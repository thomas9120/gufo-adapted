#include "src/models/qwen38_flash_next/engine.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <type_traits>

#include "src/core/gguf_reader.hpp"
#include "src/core/platform/tuning.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace gufo::models::qwen38_flash_next {
namespace {

// Larger chunks improve long-prompt throughput on native gfx1151.
// Measurements and serving limits: docs/plans/prefill-chunk-4096.md.
constexpr std::uint32_t kPrefillChunkTokens = 4096;

void AssignError(std::string* error_msg, std::string_view message) {
  if (error_msg != nullptr) {
    *error_msg = message;
  }
}

constexpr std::array<char, 8> kSessionSnapshotMagic{'Q', 'F', 'N', 'S',
                                                    'E', 'S', 'S', '1'};

/// Host-side session fields ahead of the executor payload: the token
/// history and the logits of the last token.
struct SessionSnapshotHeader {
  std::array<char, 8> magic;
  std::uint32_t version;
  std::uint32_t vocab_size;
  std::uint32_t token_count;
  std::uint32_t hidden_rows;
  std::uint64_t executor_bytes;
  MtpLengthState draft_policy;
  std::uint32_t image_identity_bytes;
  std::uint32_t policy_concurrency;
  MtpConfidenceStopState draft_stop;
  MtpLowAcceptanceState low_acceptance;
};
static_assert(std::is_trivially_copyable_v<SessionSnapshotHeader>);

std::uint64_t SessionSnapshotHostBytes(std::uint32_t token_count,
                                       std::uint32_t vocab_size,
                                       std::size_t image_bytes) {
  return sizeof(SessionSnapshotHeader) + image_bytes +
         std::uint64_t{token_count} * sizeof(std::int32_t) +
         std::uint64_t{vocab_size} * sizeof(float);
}

}  // namespace

Model::~Model() = default;

std::shared_ptr<Model> Model::Load(const std::string& model_path,
                                   const ModelOptions& options,
                                   std::string* error_msg) {
  std::shared_ptr<Model> m(new Model());
  if (options.decode_concurrency == 0 || options.decode_concurrency > 8) {
    AssignError(error_msg, "decode concurrency must be between one and eight");
    return nullptr;
  }
  if (options.max_draft_tokens == 0) {
    AssignError(error_msg, "draft token limit must be positive");
    return nullptr;
  }
  if (!options.mtp_model_path.empty() &&
      options.max_draft_tokens > kMaxMtpDraftTokens) {
    AssignError(error_msg,
                "Flash-Next MTP supports at most seven draft tokens");
    return nullptr;
  }
  m->options_ = options;
  m->reader_ = core::GgufReader::OpenFile(model_path, error_msg);
  if (!m->reader_) {
    return nullptr;
  }
  auto weights = ModelWeights::Bind(*m->reader_, error_msg);
  if (!weights) {
    return nullptr;
  }
  m->weights_ = std::make_unique<ModelWeights>(std::move(*weights));
  const Config& c = m->weights_->config;
  if (options.max_context == 0 || options.max_context > c.context_length) {
    AssignError(error_msg, "context exceeds the model's " +
                               std::to_string(c.context_length) + " tokens");
    return nullptr;
  }
  try {
    m->vision_ = qwen::vision::Encoder::Open(
        model_path, options.vision_model_path, c.hidden_size);
  } catch (const std::exception& e) {
    AssignError(error_msg, e.what());
    return nullptr;
  }
  m->tokenizer_ =
      tokenization::QwenTokenizer::CreateFromGguf(*m->reader_, error_msg);
  if (!m->tokenizer_) {
    return nullptr;
  }
  if (c.ple_layer >= 0) {
    const auto& t = m->weights_->ple_table;
    m->ngram_ = NgramTable::Open(
        m->reader_->GetMappedRegions()[t.shard].file_descriptor, t.file_offset,
        t.rows, c.ple_head_dim, t.type, error_msg);
    if (!m->ngram_) {
      return nullptr;
    }
  }
  if (!options.mtp_model_path.empty()) {
    m->mtp_reader_ =
        core::GgufReader::OpenFile(options.mtp_model_path, error_msg);
    if (!m->mtp_reader_) {
      return nullptr;
    }
    auto mtp = MtpWeights::Bind(*m->mtp_reader_, c, error_msg);
    if (!mtp) {
      return nullptr;
    }
    m->mtp_weights_ = std::make_unique<MtpWeights>(std::move(*mtp));
  }
  m->device_ = rocm::DeviceModel::Upload(*m->weights_, *m->reader_,
                                         m->mtp_weights_.get(),
                                         m->mtp_reader_.get(), error_msg);
  if (!m->device_) {
    return nullptr;
  }
  rocm::Executor::Options exec;
  exec.max_batch = m->PrefillCapacity();
  exec.max_logit_rows =
      m->mtp_weights_
          ? static_cast<std::uint32_t>(std::min<std::uint64_t>(
                exec.max_batch, std::uint64_t{options.max_draft_tokens} + 1))
          : 1;
  exec.max_speculative = exec.max_logit_rows;
  if (m->mtp_weights_ && options.draft_vocabulary == DraftVocabulary::kPrefix &&
      m->device_->output().rows > kMtpDraftVocab) {
    std::vector<std::int32_t> specials;
    for (std::uint32_t id = kMtpDraftVocab; id < m->device_->output().rows;
         ++id) {
      if (m->tokenizer_->IsSpecialToken(
              static_cast<tokenization::TokenId>(id))) {
        specials.push_back(static_cast<std::int32_t>(id));
      }
    }
    // A vocabulary with more special tokens drafts over the full head.
    if (specials.size() <= kMtpDraftMaxSpecials) {
      exec.draft_prefix = kMtpDraftVocab;
      exec.draft_specials = std::move(specials);
      m->draft_subset_ = true;
    }
  }
  m->executor_ =
      rocm::Executor::Create(*m->device_, m->ngram_.get(), exec, error_msg);
  if (!m->executor_) {
    return nullptr;
  }
  return m;
}

std::unique_ptr<Session> Model::CreateSession(core::SessionMode mode,
                                              std::uint32_t max_context,
                                              std::string* error_msg) {
  if (max_context == 0 || max_context > options_.max_context) {
    AssignError(error_msg, "session context is outside the model limits");
    return nullptr;
  }
  auto native = executor_->CreateSession(mode, max_context, error_msg);
  if (!native) {
    return nullptr;
  }
  return std::unique_ptr<Session>(
      new Session(shared_from_this(), std::move(native)));
}

std::vector<std::int32_t> Model::Tokenize(std::string_view text) const {
  std::vector<std::int32_t> out;
  for (auto id : tokenizer_->Encode(text)) {
    out.push_back(static_cast<std::int32_t>(id));
  }
  return out;
}

std::string Model::Decode(std::span<const std::int32_t> tokens) const {
  std::vector<tokenization::TokenId> ids(tokens.begin(), tokens.end());
  return tokenizer_->Decode(ids);
}

std::string Model::TokenText(std::int32_t token) const {
  return tokenizer_->DecodeTokenCopy(static_cast<tokenization::TokenId>(token));
}

std::int32_t Model::EosToken() const noexcept {
  return static_cast<std::int32_t>(tokenizer_->GetEosTokenId());
}

bool Model::IsStopToken(std::int32_t token) const noexcept {
  return tokenizer_->IsStopToken(static_cast<tokenization::TokenId>(token));
}

std::uint32_t Model::VocabSize() const noexcept {
  return weights_->config.vocab_size;
}

std::uint32_t Model::PrefillCapacity() const noexcept {
  return std::min(kPrefillChunkTokens, options_.max_context);
}

bool Model::HasMtp() const noexcept {
  return device_->has_mtp();
}

bool Model::InDraftSubset(std::int32_t token) const {
  return token >= 0 && (static_cast<std::uint32_t>(token) < kMtpDraftVocab ||
                        tokenizer_->IsSpecialToken(
                            static_cast<tokenization::TokenId>(token)));
}

std::string Model::ModelName() const {
  return std::string(reader_->GetMetadataString("general.name")
                         .value_or("Qwen3.8-Flash-Next"));
}

const Config& Model::config() const noexcept {
  return weights_->config;
}

std::size_t Model::ResidentBytes() const noexcept {
  return device_->resident_bytes() + (vision_ ? vision_->ResidentBytes() : 0);
}

std::size_t Model::SessionBytes(core::SessionMode mode,
                                std::uint32_t context) const noexcept {
  const std::size_t vision =
      vision_ ? std::size_t{context} * (config().hidden_size * sizeof(float) +
                                        3 * sizeof(std::int32_t)) +
                    64
              : 0;
  return executor_->SessionBytes(mode, context,
                                 mode == core::SessionMode::kSpeculative
                                     ? executor_->max_speculative() - 1
                                     : 0) +
         vision;
}

std::size_t Model::DeferredScratchBytes() const {
  return executor_->DeferredScratchBytes();
}

std::size_t Session::AllocatedBytes() const noexcept {
  return session_->AllocatedBytes();
}

bool Session::MtpEnabled() const noexcept {
  return session_->mtp_enabled();
}

Session::Session(std::shared_ptr<Model> model,
                 std::unique_ptr<rocm::Session> session)
    : model_(std::move(model)),
      session_(std::move(session)),
      draft_length_(model_->options_.max_draft_tokens,
                    model_->DecodeConcurrency()),
      draft_stop_(model_->options_.max_draft_tokens) {
  logits_.resize(model_->VocabSize());
}

Session::~Session() = default;

std::uint32_t Session::Position() const noexcept {
  return session_->position();
}
std::uint32_t Session::ContextSize() const noexcept {
  return session_->max_context();
}

void Session::Reset() {
  valid_ = false;
  session_->Reset();
  tokens_.clear();
  lookup_.Clear();
  ForgetDraftVocab();
  hidden_base_ = 0;
  draft_length_.Reset();
  draft_stop_.Reset();
  low_acceptance_.Reset();
  model_->executor_->MtpRewind(*session_, 0);
  valid_ = true;
}

void Session::SetCancellationCheck(std::function<bool()> check) {
  session_->SetCancellationCheck(std::move(check));
}

void Session::ConfigureVision(
    std::shared_ptr<const qwen::vision::Prompt> prompt) {
  const auto identity = prompt ? prompt->IdentityBefore(tokens_.size())
                               : std::span<const std::uint8_t>{};
  if (!tokens_.empty() &&
      !std::ranges::equal(identity, ImageIdentityBefore(tokens_.size())))
    Reset();
  const bool was_valid = valid_;
  valid_ = false;
  session_->ConfigureVision(prompt, model_->vision_,
                            model_->executor_->stream());
  image_prompt_ = std::move(prompt);
  valid_ = was_valid;
}

std::span<const std::uint8_t> Session::ImageIdentityBefore(
    std::size_t count) const {
  return image_prompt_ ? image_prompt_->IdentityBefore(count)
                       : std::span<const std::uint8_t>{};
}

std::uint32_t Session::KeptHiddenRows() const noexcept {
  return MtpEnabled()
             ? static_cast<std::uint32_t>(tokens_.size() - hidden_base_)
             : 0;
}

std::uint64_t Session::SnapshotBytes() const {
  if (!valid_)
    return 0;
  return SessionSnapshotHostBytes(static_cast<std::uint32_t>(tokens_.size()),
                                  model_->VocabSize(),
                                  ImageIdentityBefore(tokens_.size()).size()) +
         model_->executor_->SnapshotBytes(*session_, KeptHiddenRows());
}

std::unique_ptr<SessionSnapshot> Session::SaveSnapshot(
    std::string* error_msg) const {
  if (!valid_ || tokens_.empty() || tokens_.size() != session_->position() ||
      tokens_.size() > std::numeric_limits<std::uint32_t>::max()) {
    AssignError(error_msg, "snapshot needs a synced, non-empty context");
    return nullptr;
  }
  const auto token_count = static_cast<std::uint32_t>(tokens_.size());
  const auto image_identity = ImageIdentityBefore(token_count);
  const std::uint32_t hidden_rows = KeptHiddenRows();
  const std::uint64_t executor_bytes =
      model_->executor_->SnapshotBytes(*session_, hidden_rows);
  const std::uint64_t host_bytes = SessionSnapshotHostBytes(
      token_count, model_->VocabSize(), image_identity.size());
  std::unique_ptr<SessionSnapshot> snapshot(new SessionSnapshot(host_bytes));
  std::uint8_t* out = snapshot->host_.data();
  const SessionSnapshotHeader header{
      .magic = kSessionSnapshotMagic,
      .version = kSnapshotPayloadVersion,
      .vocab_size = model_->VocabSize(),
      .token_count = token_count,
      .hidden_rows = hidden_rows,
      .executor_bytes = executor_bytes,
      .draft_policy = draft_length_.State(),
      .image_identity_bytes = static_cast<std::uint32_t>(image_identity.size()),
      .policy_concurrency = model_->DecodeConcurrency(),
      .draft_stop = draft_stop_.State(),
      .low_acceptance = low_acceptance_.State(),
  };
  std::memcpy(out, &header, sizeof(header));
  out += sizeof(header);
  if (!image_identity.empty())
    std::memcpy(out, image_identity.data(), image_identity.size());
  out += image_identity.size();
  std::memcpy(out, tokens_.data(), tokens_.size() * sizeof(std::int32_t));
  out += tokens_.size() * sizeof(std::int32_t);
  std::memcpy(out, logits_.data(), logits_.size() * sizeof(float));
  out += logits_.size() * sizeof(float);
  snapshot->executor_ =
      model_->executor_->SaveHostSnapshot(*session_, hidden_rows, error_msg);
  if (!snapshot->executor_)
    return nullptr;
  snapshot->size_ += snapshot->executor_->SizeBytes();
  return snapshot;
}

bool Session::RestoreSnapshot(const SessionSnapshot& snapshot,
                              std::string* error_msg) {
  return RestoreSnapshotImpl(snapshot.host_, snapshot.executor_, error_msg);
}

bool Session::RestoreSnapshot(std::span<const std::uint8_t> payload,
                              std::string* error_msg) {
  return RestoreSnapshotImpl(payload, {}, error_msg);
}

bool Session::RestoreSnapshotImpl(
    std::span<const std::uint8_t> payload,
    const std::shared_ptr<const rocm::HostSnapshot>& snapshot,
    std::string* error_msg) {
  const auto payload_bytes =
      payload.size() + (snapshot ? snapshot->SizeBytes() : 0);
  SessionSnapshotHeader header{};
  if (payload.size() < sizeof(header)) {
    AssignError(error_msg, "session snapshot is truncated");
    return false;
  }
  std::memcpy(&header, payload.data(), sizeof(header));
  if (header.magic != kSessionSnapshotMagic ||
      header.version != kSnapshotPayloadVersion) {
    AssignError(error_msg, "session snapshot format is not supported");
    return false;
  }
  if (header.vocab_size != model_->VocabSize() || header.token_count == 0 ||
      header.policy_concurrency != model_->DecodeConcurrency() ||
      header.token_count > ContextSize() ||
      (header.image_identity_bytes != 0 && header.image_identity_bytes != 32) ||
      payload_bytes != SessionSnapshotHostBytes(header.token_count,
                                                header.vocab_size,
                                                header.image_identity_bytes) +
                           header.executor_bytes) {
    AssignError(error_msg, "session snapshot does not fit this session");
    return false;
  }
  auto restored_policy = draft_length_;
  auto restored_stop = draft_stop_;
  auto restored_low = low_acceptance_;
  if (!restored_policy.Restore(header.draft_policy) ||
      !restored_stop.Restore(header.draft_stop) ||
      !restored_low.Restore(header.low_acceptance)) {
    AssignError(error_msg, "session snapshot draft policy is invalid");
    return false;
  }
  const std::uint8_t* in = payload.data() + sizeof(header);
  std::vector<std::uint8_t> image_identity(in,
                                           in + header.image_identity_bytes);
  if (!image_identity.empty() &&
      !std::ranges::equal(image_identity,
                          ImageIdentityBefore(header.token_count))) {
    AssignError(error_msg,
                "image snapshot requires its matching prompt attachment");
    return false;
  }
  in += header.image_identity_bytes;
  std::vector<std::int32_t> tokens(header.token_count);
  std::memcpy(tokens.data(), in, tokens.size() * sizeof(std::int32_t));
  in += tokens.size() * sizeof(std::int32_t);
  std::vector<float> logits(header.vocab_size);
  std::memcpy(logits.data(), in, logits.size() * sizeof(float));
  in += logits.size() * sizeof(float);

  if (image_identity.empty() &&
      !ImageIdentityBefore(header.token_count).empty())
    ConfigureVision(nullptr);
  valid_ = false;
  rocm::Executor::SnapshotInfo info;
  const auto remaining = ContextSize() - header.token_count;
  // Keep rollback scratch for the longer of the two draft heads' choices.
  if (model_->draft_subset_)
    restored_policy.SetDraftSaving(kMtpDraftSmallHeadSavingMs);
  const auto next_drafts =
      MtpEnabled() ? restored_policy.Choose(remaining ? remaining - 1 : 0,
                                            header.token_count)
                   : 0;
  const bool restored =
      snapshot ? model_->executor_->RestoreSnapshot(*session_, snapshot, &info,
                                                    error_msg, next_drafts)
               : model_->executor_->RestoreSnapshot(
                     *session_,
                     {in, static_cast<std::size_t>(header.executor_bytes)},
                     &info, error_msg, next_drafts);
  if (!restored) {
    Reset();
    return false;
  }
  if (info.position != header.token_count ||
      info.hidden_rows != header.hidden_rows) {
    Reset();
    AssignError(error_msg, "session snapshot positions are inconsistent");
    return false;
  }
  tokens_ = std::move(tokens);
  lookup_.Clear();
  ForgetDraftVocab();
  logits_ = std::move(logits);
  anchor_candidates_valid_ = false;
  hidden_base_ = info.position - info.hidden_rows;
  draft_token_ = 0;
  draft_length_ = restored_policy;
  draft_stop_ = restored_stop;
  low_acceptance_ = restored_low;
  stats_ = {};
  valid_ = true;
  return true;
}

SessionSnapshot::SessionSnapshot(std::uint64_t host_bytes)
    : host_(host_bytes), size_(host_bytes) {}

std::span<const std::uint8_t> SessionSnapshot::bytes() const {
  std::call_once(materialize_, [&] {
    std::unique_ptr<std::uint8_t[]> data(new std::uint8_t[size_]);
    if (!CopyTo({data.get(), size_}))
      throw std::runtime_error("snapshot materialization failed");
    data_ = std::move(data);
  });
  return {data_.get(), size_};
}

bool SessionSnapshot::CopyTo(
    std::span<std::uint8_t> destination) const noexcept {
  if (destination.size() != size_)
    return false;
  try {
    std::memcpy(destination.data(), host_.data(), host_.size());
    return executor_->CopyTo(destination.subspan(host_.size()));
  } catch (...) {
    return false;
  }
}

void SessionSnapshot::StreamTo(
    const std::function<void(std::span<const std::uint8_t>)>& sink) const {
  sink(host_);
  executor_->StreamTo(sink);
}

std::uint64_t SessionSnapshot::CopiedDeviceBytes() const noexcept {
  return executor_->CopiedBytes();
}

std::uint64_t SessionSnapshot::SharedDeviceBytes() const noexcept {
  return executor_->SharedBytes();
}

bool Session::DraftReplay(std::int32_t next_token,
                          std::vector<std::int32_t>* replay,
                          std::int32_t* hidden_row,
                          std::string* error_msg) const {
  // The draft block trails the trunk: MTP position i consumes token i+1 and
  // the trunk's hidden of position i, so positions up to the current one are
  // replayed once their successor token is known. The session keeps hidden
  // rows of positions [hidden_base_, tokens_.size()).
  rocm::Executor& exec = *model_->executor_;
  const auto size = static_cast<std::uint32_t>(tokens_.size());
  const std::uint32_t mp = exec.MtpPosition(*session_);
  if (mp >= size) {
    return true;
  }
  if (mp < hidden_base_) {
    AssignError(error_msg, "draft block fell behind the kept hidden rows");
    return false;
  }
  replay->assign(tokens_.begin() + mp + 1, tokens_.end());
  replay->push_back(next_token);
  *hidden_row = static_cast<std::int32_t>(mp - hidden_base_);
  return true;
}

void Session::SyncDraftVocab() {
  if (!model_->draft_subset_ || !MtpEnabled())
    return;
  if (draft_vocab_stale_ || draft_vocab_.scanned() > tokens_.size()) {
    draft_vocab_.Reset();
    draft_vocab_stale_ = false;
  }
  draft_vocab_.CatchUp(
      tokens_, [&](std::int32_t token) { return model_->InDraftSubset(token); },
      nullptr);
}

bool Session::DraftCatchUp(std::int32_t next_token, bool propose,
                           std::string* error_msg,
                           MtpCandidateLogits* candidates,
                           bool token_with_candidates, bool draft_subset) {
  std::vector<std::int32_t> replay;
  std::int32_t hidden_row = 0;
  if (!DraftReplay(next_token, &replay, &hidden_row, error_msg))
    return false;
  if (replay.empty())
    return true;
  auto& exec = *model_->executor_;
  if (!exec.MtpForward(
          *session_, replay, hidden_row,
          {.token = propose && (candidates == nullptr || token_with_candidates)
                        ? &draft_token_
                        : nullptr,
           .candidates = candidates,
           .draft_subset = draft_subset},
          error_msg)) {
    return false;
  }
  return true;
}

bool Session::DraftCatchUpBatch(std::span<const AdvanceRequest> requests,
                                std::string* error_msg) {
  if (requests.empty())
    return true;
  auto& exec = *requests.front().session->model_->executor_;
  std::vector<std::vector<std::int32_t>> replays(requests.size());
  std::vector<rocm::Executor::MtpBatchItem> items;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    auto& session = *requests[i].session;
    if (!session.MtpEnabled())
      continue;
    std::int32_t hidden_row = 0;
    if (!session.DraftReplay(requests[i].token, &replays[i], &hidden_row,
                             error_msg))
      return false;
    if (replays[i].empty())
      continue;
    items.push_back({session.session_.get(), replays[i], hidden_row});
  }
  return items.empty() || exec.MtpForwardBatch(items, error_msg);
}

bool Session::Feed(std::span<const std::int32_t> tokens, std::string* error_msg,
                   bool prefill) {
  rocm::Executor& exec = *model_->executor_;
  for (std::size_t off = 0; off < tokens.size(); off += exec.max_batch()) {
    const std::size_t n =
        std::min<std::size_t>(exec.max_batch(), tokens.size() - off);
    const auto chunk = tokens.subspan(off, n);
    if (MtpEnabled() && !tokens_.empty() &&
        !DraftCatchUp(chunk[0], false, error_msg)) {
      return false;
    }
    const auto mode = prefill ? rocm::Executor::ForwardMode::kPrefill
                              : rocm::Executor::ForwardMode::kDecode;
    anchor_candidates_valid_ = false;
    if (!exec.Forward(*session_, chunk, 1, logits_.data(), mode, error_msg)) {
      return false;
    }
    hidden_base_ = static_cast<std::uint32_t>(
        tokens_.size() + n - std::min<std::size_t>(n, exec.max_speculative()));
    tokens_.insert(tokens_.end(), chunk.begin(), chunk.end());
  }
  return true;
}

bool Session::Sync(std::span<const std::int32_t> prompt,
                   std::string* error_msg) {
  if (prompt.empty()) {
    AssignError(error_msg, "prompt is empty");
    return false;
  }
  if (prompt.size() > ContextSize()) {
    AssignError(error_msg, "prompt exceeds the session context");
    return false;
  }
  if (!valid_)
    Reset();
  // Recurrent state cannot be rewound, so any divergence restarts the
  // session; an extension only feeds the new tail.
  std::size_t common = 0;
  while (common < tokens_.size() && common < prompt.size() &&
         tokens_[common] == prompt[common]) {
    ++common;
  }
  if (common == prompt.size() && common == tokens_.size()) {
    return true;
  }
  if (common != tokens_.size()) {
    Reset();
    common = 0;
  }
  valid_ = false;
  const bool ok = Feed(prompt.subspan(common), error_msg, true);
  valid_ = ok;
  return ok;
}

bool Session::Evaluate(std::int32_t token, std::string* error_msg) {
  if (!valid_) {
    AssignError(error_msg,
                "session needs a successful Sync after a failed operation");
    return false;
  }
  if (tokens_.size() >= ContextSize()) {
    AssignError(error_msg, "session context is full");
    return false;
  }
  valid_ = false;
  const bool ok = Feed(std::span<const std::int32_t>(&token, 1), error_msg);
  valid_ = ok;
  return ok;
}

std::uint32_t Session::MaxVerifyWidth() const noexcept {
  return MtpEnabled() ? model_->executor_->max_speculative() : 1;
}

bool Session::TeacherForce(std::span<const std::int32_t> tokens,
                           std::vector<float>* rows, std::string* error_msg,
                           bool prefill, std::size_t keep) {
  if (!valid_ || tokens_.empty()) {
    AssignError(error_msg, "teacher forcing needs a synced session");
    return false;
  }
  const std::size_t n = tokens.size();
  const std::size_t committed = keep == 0 ? n : keep;
  if (n == 0 || committed > n || (prefill && committed != n) ||
      (prefill ? MtpEnabled() : n > MaxVerifyWidth()) ||
      tokens_.size() + n > ContextSize()) {
    AssignError(error_msg, "teacher forcing width or context out of range");
    return false;
  }
  rocm::Executor& exec = *model_->executor_;
  const std::size_t vocab = model_->VocabSize();
  rows->resize(n * vocab);
  valid_ = false;
  anchor_candidates_valid_ = false;
  // The same Forward calls DecodeStep makes: a one-token decode, or a verify
  // pass with a specified kept prefix (the draft head is never consulted).
  // Prefill arithmetic is the same at every chunk width, except that past
  // indexer_top_k a chunk boundary inside an absolute four-position group
  // changes the group's sparse attention key union.
  const auto mode = prefill  ? rocm::Executor::ForwardMode::kPrefill
                    : n == 1 ? rocm::Executor::ForwardMode::kDecode
                             : rocm::Executor::ForwardMode::kVerify;
  if (!exec.Forward(*session_, tokens, static_cast<std::uint32_t>(n),
                    rows->data(), mode, error_msg) ||
      (mode == rocm::Executor::ForwardMode::kVerify &&
       !exec.Rollback(*session_, static_cast<std::uint32_t>(committed),
                      error_msg))) {
    return false;
  }
  std::copy_n(rows->data() + (committed - 1) * vocab, vocab, logits_.begin());
  tokens_.insert(tokens_.end(), tokens.begin(), tokens.begin() + committed);
  valid_ = true;
  return true;
}

struct Session::PendingDecode {
  std::vector<std::int32_t> chain;
  std::vector<MtpProposal> proposals;
  std::uint32_t base{0};
  bool speculative{false};
  bool sampled{false};
  bool gpu_greedy{false};
  bool gpu_verification{false};
  // Sampled verification reads candidate lists, so the verified rows stay on
  // the GPU; FinishDecode fetches them only for a fallback.
  bool rows_on_device{false};
  std::size_t width{0};
  std::optional<sampling::SamplerState> draft_sampler;
  std::uint64_t draft_rng{0};
  MtpCandidateLogits candidates;
  std::int32_t draft{0};
  /// Trailing chain tokens copied by prompt lookup (after the MTP drafts).
  std::uint32_t lookup{0};
  bool confidence_stop{false};  ///< greedy chain that stops on confidence
  bool chain_gate{false};       ///< one session: stop on confidence product
  bool draft_subset{false};     ///< this cycle drafts through the small head
  bool calibrated_stop{false};  ///< drafts stop on calibrated confidence
  float chain_confidence{1.0F};
};

namespace {

/// The exact target distribution of `row` from its candidate list, or
/// nothing when the list cannot prove it (then the caller uses the row).
std::optional<sampling::SamplingDistribution> FastDistribution(
    const sampling::SamplerState& sampler, const MtpCandidateLogits& list,
    std::span<const float> row) {
  return sampler.DistributionFromTop(std::span(list.logits).first(list.size),
                                     std::span(list.ids).first(list.size),
                                     row.size());
}

/// True when every finite candidate still matches `row` bit for bit: the
/// list was selected from this very row.
bool CandidatesMatch(const MtpCandidateLogits& list,
                     std::span<const float> row) {
  for (std::size_t i = 0; i < list.size; ++i) {
    if (!std::isfinite(list.logits[i])) {
      continue;
    }
    if (list.ids[i] >= row.size() ||
        std::memcmp(&list.logits[i], &row[list.ids[i]], sizeof(float)) != 0) {
      return false;
    }
  }
  return true;
}

}  // namespace

void Session::AppendDraft(PendingDecode& pending) {
  if (pending.sampled) {
    pending.proposals.push_back(SampleMtpProposal(
        pending.candidates, *pending.draft_sampler, &pending.draft_rng));
    pending.draft = static_cast<std::int32_t>(pending.proposals.back().token);
    pending.draft_sampler->Accept(pending.proposals.back().token);
  }
  pending.chain.push_back(pending.draft);
}

bool Session::AppendLookup(PendingDecode& pending, std::size_t cap) {
  if (pending.chain.size() >= cap) {
    return false;
  }
  const auto match = lookup_.Find(tokens_, pending.chain);
  if (match.length == 0) {
    return false;
  }
  const std::size_t count =
      std::min(cap - pending.chain.size(), tokens_.size() - match.start);
  for (std::size_t i = 0; i < count; ++i) {
    const std::int32_t token = tokens_[match.start + i];
    pending.chain.push_back(token);
    if (pending.sampled) {
      // A point-mass proposal: accepted with the target's p(token), and a
      // rejection resamples from the target with the token removed.
      MtpProposal proposal;
      proposal.ids[0] = static_cast<sampling::TokenId>(token);
      proposal.probabilities[0] = 1.0F;
      proposal.size = 1;
      proposal.token = proposal.ids[0];
      proposal.probability = 1.0F;
      pending.proposals.push_back(proposal);
    }
  }
  pending.lookup = static_cast<std::uint32_t>(count);
  pending.width = pending.chain.size();
  return count != 0;
}

bool Session::StopChain(PendingDecode& pending) {
  if (!pending.chain_gate)
    return false;
  pending.chain_confidence *= MtpDraftConfidence(std::span<const float>(
      pending.candidates.logits.data(), pending.candidates.size));
  return pending.chain.size() > 1 &&
         pending.chain_confidence < kMtpDraftChainConfidence;
}

bool Session::StopCalibrated(PendingDecode& pending) {
  if (!pending.calibrated_stop)
    return false;
  // Top-1 mass of this proposal. The committed accept/reject is learned
  // later, from verification, not from this confidence.
  return !draft_stop_.Keep(MtpDraftConfidence(std::span<const float>(
      pending.candidates.logits.data(), pending.candidates.size)));
}

bool Session::PrepareDecode(const DecodeRequest& request,
                            PendingDecode* pending, std::string* error_msg,
                            bool defer_head,
                            std::optional<std::uint32_t> batch_drafts) {
  const auto max_tokens = request.max_tokens;
  auto& sampler = *request.sampler;
  auto* result = request.result;
  const bool stop_at_eos = request.stop_at_eos;
  if (result == nullptr || max_tokens == 0 || tokens_.empty()) {
    AssignError(
        error_msg,
        "decode needs an output, a positive budget and a synced prompt");
    return false;
  }
  *result = {};
  const auto is_stop = [&](std::int32_t token) {
    return stop_at_eos && model_->IsStopToken(token);
  };
  rocm::Executor& exec = *model_->executor_;
  const std::size_t room = ContextSize() - tokens_.size();
  const std::size_t cap =
      std::min<std::size_t>({max_tokens, room, exec.max_speculative()});
  // A cycle that will draft small pays less per draft. The anchor is not
  // sampled yet; the window alone decides, from committed tokens only.
  if (!batch_drafts && MtpEnabled() && cap > 1) {
    SyncDraftVocab();
    draft_length_.SetDraftSaving(model_->draft_subset_ && draft_vocab_.Quiet()
                                     ? kMtpDraftSmallHeadSavingMs
                                     : 0.0F);
  }
  // Greedy output does not depend on the width. A greedy cycle that is not
  // in a jointly planned batch drafts to the limit and stops on each
  // proposal's confidence instead, alone or beside sampled requests.
  const bool confidence_stop =
      MtpEnabled() && !batch_drafts && !sampler.config().uses_random_sampling();
  // The chain gate's threshold is a one-session price. A server with more
  // sessions pays a batch's cost per draft; its greedy cycles stop on
  // calibrated confidence instead, priced at the configured sessions.
  const auto concurrency = model_->DecodeConcurrency();
  const bool chain_gate = confidence_stop && concurrency == 1;
  // Sampled output does depend on the width. Those same cycles draft to the
  // limit and stop on calibrated confidence, so one request matches itself
  // in an unplanned batch. Jointly planned batches keep their shared width.
  const bool calibrated_stop = MtpEnabled() && !batch_drafts &&
                               (sampler.config().uses_random_sampling() ||
                                (confidence_stop && !chain_gate));
  const std::size_t width =
      MtpEnabled() && cap > 1
          ? 1 + (batch_drafts ? std::min<std::uint32_t>(*batch_drafts, cap - 1)
                 : (confidence_stop || calibrated_stop)
                     ? std::min<std::uint32_t>(
                           model_->options_.max_draft_tokens,
                           static_cast<std::uint32_t>(cap - 1))
                     : draft_length_.Choose(
                           static_cast<std::uint32_t>(cap - 1),
                           static_cast<std::uint32_t>(tokens_.size())))
          : cap;
  if (width == 0) {
    result->stop = true;
    return true;
  }
  // After a sampled MTP cycle logits_ is the kept verification row, whose
  // candidate list FinishDecode left behind: sample it without a full scan.
  std::optional<sampling::TokenId> fast_anchor;
  if (anchor_candidates_valid_) {
    anchor_candidates_valid_ = false;
    if (CandidatesMatch(anchor_candidates_, logits_)) {
      fast_anchor = sampler.SampleFromTop(
          std::span(anchor_candidates_.logits).first(anchor_candidates_.size),
          std::span(anchor_candidates_.ids).first(anchor_candidates_.size),
          logits_.size());
    }
  }
  const auto anchor = static_cast<std::int32_t>(
      fast_anchor ? *fast_anchor : sampler.Sample(logits_));
  if (is_stop(anchor)) {
    result->stop = true;
    return true;
  }
  if (!MtpEnabled() || width < 2) {
    if (MtpEnabled() && !defer_head &&
        !DraftCatchUp(anchor, false, error_msg)) {
      return false;
    }
    pending->chain = {anchor};
    pending->base = static_cast<std::uint32_t>(tokens_.size());
    return true;
  }

  const std::uint32_t base = static_cast<std::uint32_t>(tokens_.size());
  const bool sampled = sampler.config().uses_random_sampling();
  const bool gpu_greedy = sampler.config().can_use_unmodified_argmax();
  const bool gpu_verification = gpu_greedy;
  pending->confidence_stop = confidence_stop;
  pending->chain_gate = chain_gate;
  pending->calibrated_stop = calibrated_stop;
  SyncDraftVocab();
  pending->draft_subset = model_->draft_subset_ &&
                          draft_vocab_.Fits(anchor, [&](std::int32_t token) {
                            return model_->InDraftSubset(token);
                          });
  if (!defer_head &&
      !DraftCatchUp(anchor, true, error_msg,
                    sampled || confidence_stop ? &pending->candidates : nullptr,
                    confidence_stop, pending->draft_subset)) {
    return false;
  }
  // A cycle-local proposal stream needs no pending RNG state in snapshots.
  // Target verification keeps its own draws after this independent seed.
  pending->draft_rng =
      sampled ? sampling::NextRandom(sampler.mutable_rng_state()) : 0;
  pending->draft_sampler = sampler;
  pending->draft_sampler->Accept(static_cast<sampling::TokenId>(anchor));
  pending->chain = {anchor};
  pending->draft = draft_token_;
  // A small-head cycle drafts cheaply enough to keep its full width.
  // Full-head cycles retain bounded recovery probes while acceptance is low.
  pending->width =
      confidence_stop && !pending->draft_subset
          ? 1 + low_acceptance_.Choose(static_cast<std::uint32_t>(width - 1))
          : width;
  pending->base = base;
  pending->speculative = true;
  pending->sampled = sampled;
  pending->gpu_greedy = gpu_greedy;
  pending->gpu_verification = gpu_verification;
  if (!gpu_verification && verify_logits_.empty()) {
    verify_logits_.resize(exec.max_speculative() * model_->VocabSize());
  }
  // ModelOptions::prompt_lookup (single session): after each kept MTP
  // draft, a long enough match in the committed tokens fills the rest of
  // the chain up to the verify width and ends drafting.
  const bool lookup =
      model_->options_.prompt_lookup && !batch_drafts && !defer_head;
  if (lookup) {
    lookup_.Extend(tokens_);
  }
  if (calibrated_stop)
    draft_stop_.Begin(base,
                      pending->draft_subset ? kMtpDraftSmallHeadSavingMs : 0.0F,
                      concurrency);
  if (!defer_head) {
    while (pending->chain.size() < pending->width) {
      if (StopChain(*pending))
        break;
      if (StopCalibrated(*pending))
        break;
      AppendDraft(*pending);
      if (lookup && AppendLookup(*pending, cap)) {
        break;
      }
      if (pending->chain.size() < pending->width &&
          !exec.MtpForward(
              *session_, std::span<const std::int32_t>(&pending->draft, 1), -1,
              {.token = sampled ? nullptr : &pending->draft,
               .candidates =
                   sampled || confidence_stop ? &pending->candidates : nullptr,
               .draft_subset = pending->draft_subset},
              error_msg)) {
        return false;
      }
    }
  }
  return true;
}

bool Session::FinishDecode(const DecodeRequest& request,
                           const PendingDecode& pending,
                           std::string* error_msg) {
  auto& sampler = *request.sampler;
  auto* result = request.result;
  auto& exec = *model_->executor_;
  const auto& chain = pending.chain;
  const auto& proposals = pending.proposals;
  const auto base = pending.base;
  const bool sampled = pending.sampled;
  const bool gpu_greedy = pending.gpu_greedy;
  const bool gpu_verification = pending.gpu_verification;
  const auto anchor = chain.front();
  const auto k = static_cast<std::uint32_t>(chain.size());
  const auto vocab = model_->VocabSize();
  const auto is_stop = [&](std::int32_t token) {
    return request.stop_at_eos && model_->IsStopToken(token);
  };
  if (!pending.speculative) {
    draft_length_.ObserveArToken();
    hidden_base_ = base;
    tokens_.push_back(anchor);
    sampler.Accept(static_cast<sampling::TokenId>(anchor));
    result->tokens.push_back(anchor);
    return true;
  }
  sampler.Accept(static_cast<sampling::TokenId>(anchor));
  std::array<rocm::ArgmaxCandidate, kMaxMtpDraftTokens> greedy{};
  if (gpu_greedy &&
      !exec.GreedyMtpPredictions(std::span(greedy).first(k - 1), error_msg)) {
    return false;
  }
  // Tuning::fast_sampling: sampled verification reads GPU-selected
  // candidate lists instead of scanning full host rows (exact, see
  // FastDistribution); without them it falls back to the rows.
  std::array<MtpCandidateLogits, kMaxMtpDraftTokens + 1> candidates{};
  std::string candidate_error;
  const bool have_candidates =
      sampled && platform::PlatformTuning().fast_sampling &&
      k <= candidates.size() && sampler.config().top_k > 0 &&
      !sampler.config().penalties_enabled() &&
      exec.VerifyCandidates(std::span(candidates).first(k), &candidate_error);
  // Full rows kept on the GPU are fetched once, only when a row is needed.
  bool rows_ready = !pending.rows_on_device;
  const auto ensure_rows = [&] {
    if (!rows_ready) {
      rows_ready =
          exec.DownloadVerification(k, verify_logits_.data(), error_msg);
    }
    return rows_ready;
  };
  if (sampled && !have_candidates && !ensure_rows()) {
    return false;
  }
  std::uint32_t keep = 1;
  std::optional<std::int32_t> correction;
  while (keep < k) {
    if (gpu_greedy) {
      const auto& prediction = greedy[keep - 1];
      if (!std::isfinite(prediction.value)) {
        AssignError(error_msg, "logit distribution contains no finite values");
        return false;
      }
      if (is_stop(prediction.index)) {
        result->stop = true;
        break;
      }
      if (prediction.index != chain[keep]) {
        break;
      }
      sampler.Accept(static_cast<sampling::TokenId>(prediction.index));
      ++keep;
      continue;
    }
    if (sampled) {
      const auto row = std::span<const float>(verify_logits_)
                           .subspan((keep - 1) * vocab, vocab);
      std::optional<sampling::SamplingDistribution> target;
      if (have_candidates) {
        target = FastDistribution(sampler, candidates[keep - 1], row);
      }
      if (!target && !ensure_rows()) {
        return false;
      }
      const auto verified =
          target
              ? VerifyMtpProposal(*target, vocab, proposals[keep - 1], sampler)
              : VerifyMtpProposal(row, proposals[keep - 1], sampler);
      const auto token = static_cast<std::int32_t>(verified.token);
      const bool accepted = verified.accepted;
      if (is_stop(token)) {
        result->stop = true;
        break;
      }
      if (!accepted) {
        correction = token;
        break;
      }
      sampler.Accept(static_cast<sampling::TokenId>(token));
      ++keep;
      continue;
    }
    const auto decision = VerifyDraft(std::span<const float>(verify_logits_)
                                          .subspan((keep - 1) * vocab, vocab),
                                      chain[keep], sampler, is_stop);
    if (decision != DraftDecision::kAccept) {
      result->stop = decision == DraftDecision::kStop;
      break;
    }
    ++keep;
  }
  if (!exec.Rollback(*session_, keep, error_msg,
                     gpu_verification || pending.rows_on_device ? logits_.data()
                                                                : nullptr)) {
    return false;
  }
  if (!gpu_verification) {
    // Rows kept on the GPU: Rollback already wrote the frontier to logits_.
    if (!pending.rows_on_device) {
      std::copy_n(verify_logits_.data() + (keep - 1) * vocab, vocab,
                  logits_.begin());
    }
    // The next PrepareDecode samples its anchor from this row's list.
    anchor_candidates_valid_ = have_candidates;
    if (have_candidates) {
      anchor_candidates_ = candidates[keep - 1];
    }
  }
  hidden_base_ = base;
  tokens_.insert(tokens_.end(), chain.begin(), chain.begin() + keep);
  result->tokens.assign(chain.begin(), chain.begin() + keep);
  stats_.cycles += 1;
  stats_.small_head_cycles += pending.draft_subset;
  stats_.drafted += k - 1;
  stats_.accepted += keep - 1;
  // Copied proposals trail the MTP drafts; the length controller sees only
  // the MTP part.
  const std::uint32_t mtp_drafts = k - 1 - pending.lookup;
  const std::uint32_t mtp_accepted = std::min(keep - 1, mtp_drafts);
  stats_.lookup += pending.lookup;
  stats_.lookup_accepted += keep - 1 - mtp_accepted;
  // A target stop ends the request; it does not classify the remaining
  // proposals as failed predictions. Sampled calibration sees that same
  // committed prefix: accepts, then the first stochastic rejection.
  draft_length_.Observe(mtp_accepted, result->stop ? mtp_accepted : mtp_drafts,
                        base);
  if (pending.calibrated_stop)
    draft_stop_.Observe(mtp_accepted, result->stop ? mtp_accepted : mtp_drafts);
  if (pending.confidence_stop)
    low_acceptance_.Observe(mtp_accepted, !pending.draft_subset);

  // The next call knows the next sampled anchor. Defer draft catch-up until
  // then, retaining this session's target hidden rows across interleaving.
  exec.MtpRewind(*session_, base);
  if (correction) {
    // Evaluate the residual as the next cycle's anchor, avoiding a separate
    // target pass. Preserve the actual draw: resampling p would be biased.
    sampler.DeferSample(static_cast<sampling::TokenId>(*correction));
  }
  return true;
}

bool Session::DecodeStep(std::size_t max_tokens,
                         sampling::SamplerState& sampler, DecodeResult* result,
                         std::string* error_msg, bool stop_at_eos) {
  if (!valid_) {
    AssignError(error_msg,
                "session needs a successful Sync after a failed operation");
    return false;
  }
  valid_ = false;
  const DecodeRequest request{this, max_tokens, &sampler, result, stop_at_eos};
  PendingDecode pending;
  if (!PrepareDecode(request, &pending, error_msg)) {
    return false;
  }
  if (pending.chain.empty()) {
    valid_ = true;
    return true;
  }
  // Tuning::fast_sampling: sampled verification rows stay on the GPU; the
  // candidate lists replace them unless a row is needed (DownloadVerification).
  pending.rows_on_device =
      pending.speculative && !pending.gpu_verification && pending.sampled &&
      platform::PlatformTuning().fast_sampling && sampler.config().top_k > 0 &&
      !sampler.config().penalties_enabled();
  float* logits = !pending.speculative ? logits_.data()
                  : pending.gpu_verification || pending.rows_on_device
                      ? nullptr
                      : verify_logits_.data();
  if (!model_->executor_->Forward(
          *session_, pending.chain, pending.chain.size(), logits,
          pending.speculative ? rocm::Executor::ForwardMode::kVerify
                              : rocm::Executor::ForwardMode::kDecode,
          error_msg, pending.rows_on_device)) {
    return false;
  }
  const bool ok = FinishDecode(request, pending, error_msg);
  valid_ = ok;
  return ok;
}

template<class Request>
bool Session::RunIsolatedBatch(std::span<const Request> requests,
                               std::string* error_msg) {
  if (requests.empty() || requests.size() > 8) {
    AssignError(error_msg, "batch must contain 1..8 sessions");
    return false;
  }
  std::array<BatchOutcome, 8> outcomes{};
  std::array<std::uint64_t, 8> epochs{};
  std::array<sampling::SamplerState::DrawState, 8> draws{};
  std::vector<Request> active;
  active.reserve(requests.size());
  Model* model = nullptr;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    auto& outcome = outcomes[i];
    if (r.session)
      epochs[i] = r.session->session_->MutationEpoch();
    bool valid = r.session && r.session->valid_;
    if constexpr (std::is_same_v<Request, DecodeRequest>) {
      valid = valid && r.sampler && r.result && r.max_tokens > 0 &&
              !r.session->tokens_.empty();
    } else {
      valid = valid && r.token >= 0 &&
              static_cast<std::uint32_t>(r.token) <
                  r.session->model_->VocabSize() &&
              r.session->Position() < r.session->ContextSize();
    }
    for (std::size_t j = 0; j < requests.size(); ++j) {
      if (i == j)
        continue;
      valid = valid && r.session != requests[j].session;
      if constexpr (std::is_same_v<Request, DecodeRequest>)
        valid = valid && r.sampler != requests[j].sampler &&
                r.result != requests[j].result;
    }
    if (valid && model && r.session->model_.get() != model)
      valid = false;
    if (!valid) {
      outcome.error = "invalid or non-independent batch request";
      continue;
    }
    if (!r.session->session_->CheckCancellation(&outcome.error))
      continue;
    model = r.session->model_.get();
    if constexpr (std::is_same_v<Request, DecodeRequest>) {
      draws[i] = r.sampler->SaveDrawState();
    }
    auto copy = r;
    copy.outcome = &outcome;
    active.push_back(copy);
  }
  std::string shared_error;
  try {
    if (!active.empty()) {
      if constexpr (std::is_same_v<Request, DecodeRequest>)
        (void)DecodeBatchImpl(active, &shared_error);
      else
        (void)EvaluateBatchImpl(active, &shared_error);
    }
  } catch (const std::exception& exception) {
    shared_error = exception.what();
  }
  bool success = true;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    auto& outcome = outcomes[i];
    if (!outcome.completed && outcome.error.empty()) {
      auto& session = *r.session;
      if (session.session_->Cancelled()) {
        outcome.error = "generation cancelled";
      } else if (session.session_->MutationEpoch() == epochs[i]) {
        // A preparation/batch-allocation failure did not touch this peer.
        // Retry it independently, without changing its cached frontier.
        session.valid_ = true;
        if constexpr (std::is_same_v<Request, DecodeRequest>) {
          r.sampler->RestoreDrawState(draws[i]);
        }
        try {
          if constexpr (std::is_same_v<Request, DecodeRequest>)
            outcome.completed =
                session.DecodeStep(r.max_tokens, *r.sampler, r.result,
                                   &outcome.error, r.stop_at_eos);
          else
            outcome.completed = session.Evaluate(r.token, &outcome.error);
        } catch (const std::exception& exception) {
          outcome.error = exception.what();
        }
      } else {
        outcome.error =
            shared_error.empty() ? "batch execution failed" : shared_error;
      }
    }
    if (!outcome.completed) {
      if (outcome.error.empty())
        outcome.error = "batch request failed";
      if (r.session && r.session->session_->MutationEpoch() != epochs[i])
        r.session->valid_ = false;
      if (success)
        AssignError(error_msg, outcome.error);
      success = false;
    } else {
      r.session->valid_ = true;
    }
    if (r.outcome)
      *r.outcome = std::move(outcome);
  }
  return success;
}

bool Session::DecodeBatch(std::span<const DecodeRequest> requests,
                          std::string* error_msg) {
  return RunIsolatedBatch(requests, error_msg);
}

bool Session::EvaluateBatch(std::span<const AdvanceRequest> requests,
                            std::string* error_msg) {
  return RunIsolatedBatch(requests, error_msg);
}

bool Session::DecodeBatchImpl(std::span<const DecodeRequest> requests,
                              std::string* error_msg) {
  if (requests.size() == 1) {
    const auto& r = requests.front();
    r.outcome->completed = r.session->DecodeStep(
        r.max_tokens, *r.sampler, r.result, &r.outcome->error, r.stop_at_eos);
    return r.outcome->completed;
  }
  auto& exec = *requests.front().session->model_->executor_;
  std::optional<std::uint32_t> batch_drafts;
  std::uint32_t batch_context = 0;
  auto& policy = requests.front().session->model_->batch_policy_;
  if (std::ranges::all_of(
          requests, [](const auto& r) { return r.session->MtpEnabled(); }) &&
      std::ranges::none_of(requests, [](const auto& request) {
        return request.sampler->config().uses_random_sampling();
      })) {
    std::array<MtpBatchController::Row, 8> rows{};
    for (std::size_t i = 0; i < requests.size(); ++i) {
      const auto& r = requests[i];
      const auto cap = std::min<std::size_t>(
          r.max_tokens, r.session->ContextSize() - r.session->Position());
      rows[i] = {&r.session->draft_length_,
                 static_cast<std::uint32_t>(
                     std::min<std::size_t>(cap, exec.max_speculative())) -
                     (cap != 0)};
      batch_context = std::max(batch_context, r.session->Position());
    }
    batch_drafts =
        policy.Choose(std::span(rows).first(requests.size()), batch_context);
  }
  const auto cycle_start = std::chrono::steady_clock::now();
  std::vector<PendingDecode> pending(requests.size());
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    try {
      if (!r.session->PrepareDecode(r, &pending[i], &r.outcome->error, true,
                                    batch_drafts))
        pending[i] = {};
    } catch (const std::exception& exception) {
      r.outcome->error = exception.what();
      pending[i] = {};
    }
  }
  std::vector<AdvanceRequest> catchup;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    if (!pending[i].chain.empty())
      catchup.push_back({requests[i].session, pending[i].chain.front()});
  }
  if (!DraftCatchUpBatch(catchup, error_msg))
    return false;
  // Each round shares predictor projections across ready sessions.
  // Attention state, proposal distributions and RNG streams stay private.
  for (;;) {
    std::vector<rocm::Executor::MtpHeadItem> heads;
    std::vector<rocm::Executor::MtpBatchItem> bodies;
    for (std::size_t i = 0; i < requests.size(); ++i) {
      auto& p = pending[i];
      if (!requests[i].session->session_->Cancelled() && p.speculative &&
          p.chain.size() < p.width) {
        heads.push_back(
            {requests[i].session->session_.get(),
             {.token = p.sampled ? nullptr : &p.draft,
              .candidates =
                  p.sampled || p.confidence_stop ? &p.candidates : nullptr,
              .draft_subset = p.draft_subset}});
      }
    }
    if (heads.empty())
      break;
    if (!exec.MtpHeads(heads, error_msg))
      return false;
    for (std::size_t i = 0; i < requests.size(); ++i) {
      auto& p = pending[i];
      if (requests[i].session->session_->Cancelled() || !p.speculative ||
          p.chain.size() >= p.width)
        continue;
      if (StopChain(p)) {
        p.width = p.chain.size();
        continue;
      }
      if (requests[i].session->StopCalibrated(p)) {
        p.width = p.chain.size();
        continue;
      }
      AppendDraft(p);
      if (p.chain.size() < p.width)
        bodies.push_back({requests[i].session->session_.get(),
                          std::span<const std::int32_t>(&p.draft, 1), -1});
    }
    if (!bodies.empty() && !exec.MtpForwardBatch(bodies, error_msg))
      return false;
  }
  std::vector<rocm::Executor::BatchItem> items;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    if (!r.session->session_->Cancelled() && !pending[i].chain.empty()) {
      items.push_back({r.session->session_.get(), pending[i].chain,
                       pending[i].speculative});
    }
  }
  if (!items.empty() && !exec.ForwardBatch(items, error_msg))
    return false;
  std::uint32_t offset = 0;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& p = pending[i];
    const auto& r = requests[i];
    auto& session = *r.session;
    const bool included =
        std::any_of(items.begin(), items.end(), [&](const auto& item) {
          return item.session == session.session_.get();
        });
    const auto row_offset = offset;
    if (included)
      offset += p.chain.size();
    if (!r.outcome->error.empty())
      continue;
    if (session.session_->Cancelled()) {
      r.outcome->error = "generation cancelled";
      continue;
    }
    if (p.chain.empty()) {
      r.outcome->completed = true;
      continue;
    }
    session.anchor_candidates_valid_ = false;
    float* logits = !p.speculative       ? session.logits_.data()
                    : p.gpu_verification ? nullptr
                                         : session.verify_logits_.data();
    try {
      r.outcome->completed =
          exec.SelectBatchLogits(row_offset, p.chain.size(), logits,
                                 &r.outcome->error) &&
          session.FinishDecode(r, p, &r.outcome->error);
    } catch (const std::exception& exception) {
      r.outcome->error = exception.what();
    }
  }
  if (batch_drafts && items.size() == requests.size() &&
      std::ranges::all_of(requests,
                          [](const auto& r) { return r.outcome->completed; })) {
    const auto ms = std::chrono::duration<float, std::milli>(
                        std::chrono::steady_clock::now() - cycle_start)
                        .count();
    policy.Observe(requests.size(), batch_context, *batch_drafts, ms);
  }
  return true;
}

bool Session::EvaluateBatchImpl(std::span<const AdvanceRequest> requests,
                                std::string* error_msg) {
  if (requests.size() == 1) {
    const auto& r = requests.front();
    r.outcome->completed = r.session->Evaluate(r.token, &r.outcome->error);
    return r.outcome->completed;
  }
  auto& exec = *requests.front().session->model_->executor_;
  if (!DraftCatchUpBatch(requests, error_msg))
    return false;
  std::vector<rocm::Executor::BatchItem> items;
  for (const auto& r : requests) {
    items.push_back({r.session->session_.get(), {&r.token, 1}, false});
  }
  if (!exec.ForwardBatch(items, error_msg)) {
    return false;
  }
  for (std::size_t i = 0; i < requests.size(); ++i) {
    auto& session = *requests[i].session;
    auto& outcome = *requests[i].outcome;
    if (session.session_->Cancelled()) {
      outcome.error = "generation cancelled";
      continue;
    }
    try {
      session.anchor_candidates_valid_ = false;
      if (!exec.SelectBatchLogits(i, 1, session.logits_.data(), &outcome.error))
        continue;
      session.hidden_base_ = static_cast<std::uint32_t>(session.tokens_.size());
      session.tokens_.push_back(requests[i].token);
      outcome.completed = true;
    } catch (const std::exception& exception) {
      outcome.error = exception.what();
    }
  }
  return true;
}

}  // namespace gufo::models::qwen38_flash_next
