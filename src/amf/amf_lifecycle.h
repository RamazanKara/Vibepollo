/**
 * @file src/amf/amf_lifecycle.h
 * @brief Small, driver-independent AMF lifecycle primitives.
 *
 * These helpers intentionally contain no AMF or D3D types. Production uses them
 * for input ownership, bounded retry, and teardown; tests can therefore exercise
 * the same state transitions with a fake AMF implementation.
 */
#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <optional>
#include <thread>
#include <utility>

namespace amf::lifecycle {

  /**
   * @brief Ownership state for a reusable native AMF input surface.
   */
  enum class input_surface_state_e {
    free,  ///< Available to the converter.
    reserved,  ///< Reserved for conversion but not submitted.
    in_flight  ///< Accepted and still owned by AMF.
  };

  /**
   * @brief Driver-independent state associated with one input-surface slot.
   */
  struct input_surface_state_t {
    uint64_t frame_index = 0;  ///< Frame index currently associated with the slot.
    input_surface_state_e state = input_surface_state_e::free;  ///< Current ownership state.
    bool release_notified = false;  ///< Whether AMF already issued its release callback.
  };

  /**
   * @brief Check whether an AMF rate-control mode requires PreAnalysis.
   *
   * @param mode AMF rate-control enum value.
   * @return True for QVBR, HQVBR, and HQCBR.
   */
  inline bool rate_control_requires_preanalysis(int mode) noexcept {
    // AMF uses these values consistently for AVC, HEVC, and AV1.
    return mode >= 4 && mode <= 6;  // QVBR, HQVBR, HQCBR
  }

  // AMD documents a lookahead depth of one for the ultra-low-latency usage
  // preset. Keep native streaming on that bounded pipeline instead of inheriting
  // the 11-frame default used by transcoding/high-quality presets.
  inline constexpr int low_latency_preanalysis_lookahead_depth = 1;  ///< ULL PreAnalysis depth documented by AMD.

  /**
   * @brief Resolved PreAnalysis state for one encoder configuration.
   */
  struct preanalysis_plan_t {
    bool enabled = false;  ///< Whether PreAnalysis must be enabled.
    int lookahead_depth = 0;  ///< Number of delayed input frames.
    bool enabled_for_rate_control = false;  ///< Whether rate control required PreAnalysis.
  };

  /**
   * @brief Resolve explicit and rate-control-implied PreAnalysis settings.
   *
   * @param rate_control Requested AMF rate-control mode.
   * @param explicit_preanalysis Explicit user setting.
   * @return Resolved PreAnalysis plan.
   */
  inline preanalysis_plan_t resolve_preanalysis(const std::optional<int> &rate_control,
                                                const std::optional<int> &explicit_preanalysis) noexcept {
    const bool required_by_rate_control = rate_control && rate_control_requires_preanalysis(*rate_control);
    const bool explicitly_enabled = explicit_preanalysis && *explicit_preanalysis != 0;
    const bool enabled = required_by_rate_control || explicitly_enabled;
    return {
      enabled,
      enabled ? low_latency_preanalysis_lookahead_depth : 0,
      required_by_rate_control,
    };
  }

  /**
   * @brief Apply rate control and dependent PreAnalysis properties in AMD's required order.
   *
   * @tparam ApplyRateControl Rate-control property callback type.
   * @tparam ApplyPreanalysis PreAnalysis-enable callback type.
   * @tparam ApplyLookahead Lookahead-depth callback type.
   * @param rate_control Requested AMF rate-control mode.
   * @param configure_preanalysis Whether the PreAnalysis property should be written.
   * @param plan Resolved PreAnalysis plan.
   * @param lookahead_depth Requested lookahead depth.
   * @param apply_rate_control Callback that writes rate control.
   * @param apply_preanalysis Callback that writes the PreAnalysis state.
   * @param apply_lookahead Callback that writes the lookahead depth.
   * @return True when every requested property was applied.
   */
  template<typename ApplyRateControl, typename ApplyPreanalysis, typename ApplyLookahead>
  bool apply_rate_control_and_preanalysis(
    const std::optional<int> &rate_control,
    bool configure_preanalysis,
    const preanalysis_plan_t &plan,
    int lookahead_depth,
    ApplyRateControl &&apply_rate_control,
    ApplyPreanalysis &&apply_preanalysis,
    ApplyLookahead &&apply_lookahead) {
    if (rate_control && !apply_rate_control(*rate_control)) {
      return false;
    }
    if (configure_preanalysis && !apply_preanalysis(plan.enabled)) {
      return false;
    }
    if (plan.enabled && !apply_lookahead(std::max(1, lookahead_depth))) {
      return false;
    }
    return true;
  }

  /**
   * @brief Determine whether delayed output is expected after accepted inputs.
   *
   * @param accepted_without_output Accepted input count since the last output.
   * @param lookahead_depth Configured lookahead depth.
   * @return True once the encoder has more inputs than its lookahead retains.
   */
  inline bool delayed_output_is_expected(int accepted_without_output, int lookahead_depth) noexcept {
    return accepted_without_output > std::max(0, lookahead_depth);
  }

  /**
   * @brief Resolve Sunshine's H.264 coder option to AMF's CABAC boolean.
   *
   * @param coder_mode Zero automatic, one CABAC, or two CAVLC.
   * @return CABAC boolean, or no value to preserve the driver default.
   */
  inline std::optional<int> resolve_h264_cabac(int coder_mode) noexcept {
    if (coder_mode == 1) return 1;  // CABAC
    if (coder_mode == 2) return 0;  // CAVLC
    return std::nullopt;  // auto: preserve the driver default
  }

  /**
   * @brief Select a free surface when repeating or reconverting an input frame.
   *
   * @tparam Slot Slot state type.
   * @tparam SlotCount Number of slots in the ring.
   * @param slots Surface ring.
   * @param source_slot Previously rendered source slot.
   * @param next_slot Preferred rotation start.
   * @return Free slot index, or no value when every surface is AMF-owned.
   */
  template<typename Slot, std::size_t SlotCount>
  std::optional<std::size_t> select_repeat_surface(
    const std::array<Slot, SlotCount> &slots,
    std::size_t source_slot,
    std::size_t next_slot) noexcept {
    static_assert(SlotCount > 0);
    if (source_slot >= SlotCount) {
      return std::nullopt;
    }
    if (slots[source_slot].state == input_surface_state_e::free) {
      return source_slot;
    }
    for (std::size_t offset = 0; offset < SlotCount; ++offset) {
      const auto candidate = (next_slot + offset) % SlotCount;
      if (slots[candidate].state == input_surface_state_e::free) {
        return candidate;
      }
    }
    return std::nullopt;
  }

  /**
   * @brief Apply AMF's surface-release callback to a slot.
   *
   * @param slot Surface state receiving the callback.
   * @return True when the slot became immediately reusable.
   */
  inline bool on_surface_released(input_surface_state_t &slot) noexcept {
    slot.release_notified = true;
    if (slot.state != input_surface_state_e::in_flight) {
      return false;
    }

    slot.state = input_surface_state_e::free;
    slot.frame_index = 0;
    return true;
  }

  /**
   * @brief Commit surface ownership after SubmitInput accepts it.
   *
   * @param slot Accepted surface state.
   * @param frame_index Input frame index.
   * @return True when a synchronous release made the slot immediately reusable.
   */
  inline bool on_input_accepted(input_surface_state_t &slot, uint64_t frame_index) noexcept {
    slot.state = input_surface_state_e::in_flight;
    slot.frame_index = frame_index;
    if (!slot.release_notified) {
      return false;
    }

    slot.state = input_surface_state_e::free;
    slot.frame_index = 0;
    return true;
  }

  /**
   * @brief Commit LTR/RFI bookkeeping only after AMF accepts the input.
   *
   * @tparam SlotCount Size of the LTR state arrays.
   * @tparam FlagRecoveredFrame Recovery-flag callback type.
   * @param input_accepted Whether AMF accepted the current input.
   * @param effective_slots Number of enabled LTR slots.
   * @param reset_cache Whether stale slot state must be cleared.
   * @param slot_to_preserve Slot retained while clearing.
   * @param slot_to_mark Slot marked by this frame.
   * @param next_mark_slot Next rotating slot index.
   * @param consume_pending_rfi Whether this frame consumes an RFI request.
   * @param frame_after_rfi Whether the emitted frame should carry an RFI flag.
   * @param frame_index Current input frame index.
   * @param slots_valid Validity flags for LTR slots.
   * @param slot_frame_indices Frame indices stored in LTR slots.
   * @param current_mark_slot Current rotating mark slot.
   * @param rfi_pending Pending-RFI state.
   * @param flag_recovered_frame Callback that tags the accepted recovery frame.
   */
  template<std::size_t SlotCount, typename FlagRecoveredFrame>
  void commit_recovery_state(bool input_accepted,
                             int effective_slots,
                             bool reset_cache,
                             int slot_to_preserve,
                             int slot_to_mark,
                             int next_mark_slot,
                             bool consume_pending_rfi,
                             bool frame_after_rfi,
                             uint64_t frame_index,
                             std::array<bool, SlotCount> &slots_valid,
                             std::array<uint64_t, SlotCount> &slot_frame_indices,
                             int &current_mark_slot,
                             bool &rfi_pending,
                             FlagRecoveredFrame &&flag_recovered_frame) {
    if (!input_accepted) {
      return;
    }

    const int bounded_slots = std::min(effective_slots, static_cast<int>(SlotCount));
    if (reset_cache) {
      for (int slot = 0; slot < bounded_slots; ++slot) {
        if (slot != slot_to_preserve) {
          slots_valid[slot] = false;
          slot_frame_indices[slot] = 0;
        }
      }
    }
    if (slot_to_mark >= 0 && slot_to_mark < bounded_slots) {
      slots_valid[slot_to_mark] = true;
      slot_frame_indices[slot_to_mark] = frame_index;
      current_mark_slot = next_mark_slot;
    }
    if (consume_pending_rfi) {
      rfi_pending = false;
    }
    if (frame_after_rfi) {
      flag_recovered_frame(frame_index);
    }
  }

  /**
   * @brief Retry one unchanged submission while bounded backpressure persists.
   *
   * @tparam Submit Submission callback type.
   * @tparam WaitForProgress Progress-wait callback type.
   * @tparam Retryable Result-classification callback type.
   * @param submit Callback that retries the same input.
   * @param wait_for_progress Callback that waits for output or abort state.
   * @param retryable Callback identifying backpressure results.
   * @param max_retries Maximum retry count.
   * @return Final submission result.
   */
  template<typename Submit, typename WaitForProgress, typename Retryable>
  auto submit_with_bounded_retry(Submit &&submit,
                                 WaitForProgress &&wait_for_progress,
                                 Retryable &&retryable,
                                 int max_retries) {
    auto result = submit();
    for (int retry = 0; retry < max_retries && retryable(result); ++retry) {
      // A true result asks us to abort (fatal state/shutdown). The caller owns
      // policy for the still-retryable result.
      if (wait_for_progress()) {
        break;
      }
      result = submit();
    }
    return result;
  }

  /**
   * @brief Run potentially blocking driver cleanup on an owning worker.
   *
   * @tparam Work Cleanup callback type.
   * @tparam Rep Timeout representation type.
   * @tparam Period Timeout period type.
   * @param work Cleanup callback.
   * @param timeout Maximum wait before the worker is detached.
   * @return True when cleanup completed within the timeout.
   */
  template<typename Work, typename Rep, typename Period>
  bool run_with_timeout(Work &&work, std::chrono::duration<Rep, Period> timeout) {
    std::promise<void> done;
    auto done_future = done.get_future();
    std::thread worker {
      [work = std::forward<Work>(work), done = std::move(done)]() mutable {
        try {
          work();
        } catch (...) {
          // Cleanup is best-effort; completion still releases the waiter.
        }
        try {
          done.set_value();
        } catch (...) {
          // The waiter may already have abandoned the shared state.
        }
      }
    };

    if (done_future.wait_for(timeout) == std::future_status::ready) {
      worker.join();
      return true;
    }

    worker.detach();
    return false;
  }

}  // namespace amf::lifecycle
