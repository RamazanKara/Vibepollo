/**
 * @file tests/unit/platform/windows/test_native_amf_review.cpp
 * @brief Behavioral tests for native AMF ownership and lifecycle policy.
 */

#include "src/amf/amf_lifecycle.h"

#include <array>
#include <atomic>
#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#ifndef SUNSHINE_AMF_LIFECYCLE_STANDALONE
  #include "../../../tests_common.h"
#endif

using namespace std::chrono_literals;

namespace {

  enum class fake_amf_result_e {
    ok,
    input_full,
    failed
  };

  bool synchronous_release_during_submit_is_reentrant_safe() {
    std::mutex state_mutex;
    amf::lifecycle::input_surface_state_t slot;
    slot.state = amf::lifecycle::input_surface_state_e::reserved;

    bool observer_acquired_state = false;
    const auto result = amf::lifecycle::submit_with_bounded_retry(
      [&]() {
        // Fake AMF invokes OnSurfaceDataRelease synchronously from SubmitInput.
        if (state_mutex.try_lock()) {
          observer_acquired_state = true;
          amf::lifecycle::on_surface_released(slot);
          state_mutex.unlock();
        }
        return fake_amf_result_e::ok;
      },
      []() { return false; },
      [](fake_amf_result_e value) { return value == fake_amf_result_e::input_full; },
      20);

    std::lock_guard lock(state_mutex);
    const bool immediately_reusable = amf::lifecycle::on_input_accepted(slot, 7);
    return result == fake_amf_result_e::ok && observer_acquired_state && immediately_reusable &&
           slot.state == amf::lifecycle::input_surface_state_e::free && slot.frame_index == 0;
  }

  bool backpressure_retries_the_same_submission_until_accepted() {
    const std::vector<fake_amf_result_e> responses {
      fake_amf_result_e::input_full,
      fake_amf_result_e::input_full,
      fake_amf_result_e::ok,
    };
    std::size_t submit_count = 0;
    int wait_count = 0;
    const auto result = amf::lifecycle::submit_with_bounded_retry(
      [&]() { return responses.at(submit_count++); },
      [&]() {
        ++wait_count;
        return false;
      },
      [](fake_amf_result_e value) { return value == fake_amf_result_e::input_full; },
      20);
    return result == fake_amf_result_e::ok && submit_count == 3 && wait_count == 2;
  }

  bool recovery_state_changes_only_after_accepted_input() {
    std::array<bool, 4> slots_valid {true, true, false, false};
    std::array<uint64_t, 4> slot_frames {10, 20, 0, 0};
    int current_slot = 1;
    bool rfi_pending = true;
    uint64_t flagged_frame = 0;

    auto commit = [&](bool accepted) {
      amf::lifecycle::commit_recovery_state(
        accepted,
        2,
        true,
        0,
        -1,
        1,
        true,
        true,
        42,
        slots_valid,
        slot_frames,
        current_slot,
        rfi_pending,
        [&](uint64_t frame_index) { flagged_frame = frame_index; });
    };

    commit(false);
    const bool unchanged_while_rejected = slots_valid[0] && slots_valid[1] &&
                                          slot_frames[0] == 10 && slot_frames[1] == 20 &&
                                          rfi_pending && flagged_frame == 0;
    commit(true);
    return unchanged_while_rejected && slots_valid[0] && !slots_valid[1] &&
           slot_frames[0] == 10 && slot_frames[1] == 0 && !rfi_pending && flagged_frame == 42;
  }

  bool preanalysis_dependent_rate_control_is_planned_natively() {
    const auto normal = amf::lifecycle::resolve_preanalysis(3, 0);
    const auto explicit_pa = amf::lifecycle::resolve_preanalysis(3, 1);
    const auto qvbr = amf::lifecycle::resolve_preanalysis(4, 0);
    const auto hqvbr = amf::lifecycle::resolve_preanalysis(5, 0);
    const auto hqcbr = amf::lifecycle::resolve_preanalysis(6, 0);
    std::vector<int> property_order;
    const bool applied = amf::lifecycle::apply_rate_control_and_preanalysis(
      4,
      true,
      qvbr,
      qvbr.lookahead_depth,
      [&](int mode) {
        property_order.push_back(100 + mode);
        return true;
      },
      [&](bool enabled) {
        property_order.push_back(200 + static_cast<int>(enabled));
        return true;
      },
      [&](int depth) {
        property_order.push_back(300 + depth);
        return true;
      });

    return !normal.enabled && normal.lookahead_depth == 0 &&
           explicit_pa.enabled && explicit_pa.lookahead_depth == 1 && !explicit_pa.enabled_for_rate_control &&
           qvbr.enabled && qvbr.lookahead_depth == 1 && qvbr.enabled_for_rate_control &&
           hqvbr.enabled && hqvbr.lookahead_depth == 1 && hqvbr.enabled_for_rate_control &&
           hqcbr.enabled && hqcbr.lookahead_depth == 1 && hqcbr.enabled_for_rate_control &&
           applied && property_order == std::vector<int> {104, 201, 301};
  }

  bool preanalysis_pipeline_primes_and_drains_in_order() {
    struct fake_delayed_encoder_t {
      explicit fake_delayed_encoder_t(int depth):
          lookahead_depth(depth) {
      }

      std::optional<uint64_t> submit(uint64_t frame_index) {
        pending.push_back(frame_index);
        if (!amf::lifecycle::delayed_output_is_expected(
              static_cast<int>(pending.size()),
              lookahead_depth)) {
          return std::nullopt;
        }
        const auto output = pending.front();
        pending.pop_front();
        return output;
      }

      std::vector<uint64_t> drain() {
        std::vector<uint64_t> output;
        while (!pending.empty()) {
          output.push_back(pending.front());
          pending.pop_front();
        }
        return output;
      }

      int lookahead_depth;
      std::deque<uint64_t> pending;
    };

    fake_delayed_encoder_t encoder {amf::lifecycle::low_latency_preanalysis_lookahead_depth};
    const auto first = encoder.submit(100);
    const auto second = encoder.submit(101);
    const auto tail = encoder.drain();
    return !first && second && *second == 100 && tail == std::vector<uint64_t> {101} &&
           !amf::lifecycle::delayed_output_is_expected(1, 1) &&
           amf::lifecycle::delayed_output_is_expected(2, 1) &&
           amf::lifecycle::delayed_output_is_expected(1, 0);
  }

  bool automatic_h264_coder_preserves_driver_default() {
    const auto automatic = amf::lifecycle::resolve_h264_cabac(0);
    const auto cabac = amf::lifecycle::resolve_h264_cabac(1);
    const auto cavlc = amf::lifecycle::resolve_h264_cabac(2);
    return !automatic && cabac && *cabac == 1 && cavlc && *cavlc == 0;
  }

  bool repeated_input_rotates_away_from_a_lookahead_owned_surface() {
    std::array<amf::lifecycle::input_surface_state_t, 3> slots;
    slots[0].state = amf::lifecycle::input_surface_state_e::in_flight;
    slots[1].state = amf::lifecycle::input_surface_state_e::free;
    slots[2].state = amf::lifecycle::input_surface_state_e::in_flight;
    const auto rotated = amf::lifecycle::select_repeat_surface(slots, 0, 1);

    slots[0].state = amf::lifecycle::input_surface_state_e::free;
    const auto reused = amf::lifecycle::select_repeat_surface(slots, 0, 2);

    for (auto &slot : slots) slot.state = amf::lifecycle::input_surface_state_e::in_flight;
    const auto unavailable = amf::lifecycle::select_repeat_surface(slots, 0, 1);
    return rotated && *rotated == 1 && reused && *reused == 0 && !unavailable;
  }

  bool teardown_timeout_returns_control_before_a_wedged_destructor() {
    struct slow_resource_t {
      std::atomic<bool> *destroyed;
      ~slow_resource_t() {
        std::this_thread::sleep_for(150ms);
        destroyed->store(true, std::memory_order_release);
      }
    };

    std::atomic<bool> destroyed {false};
    auto resource = std::make_unique<slow_resource_t>();
    resource->destroyed = &destroyed;
    const auto start = std::chrono::steady_clock::now();
    const bool completed = amf::lifecycle::run_with_timeout(
      [resource = std::move(resource)]() mutable { resource.reset(); },
      5ms);
    const auto elapsed = std::chrono::steady_clock::now() - start;

    const auto cleanup_deadline = std::chrono::steady_clock::now() + 1s;
    while (!destroyed.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < cleanup_deadline) {
      std::this_thread::sleep_for(1ms);
    }
    return !completed && elapsed < 100ms && destroyed.load(std::memory_order_acquire);
  }

}  // namespace

#ifdef SUNSHINE_AMF_LIFECYCLE_STANDALONE

int main() {
  return synchronous_release_during_submit_is_reentrant_safe() &&
             backpressure_retries_the_same_submission_until_accepted() &&
             recovery_state_changes_only_after_accepted_input() &&
             preanalysis_dependent_rate_control_is_planned_natively() &&
             preanalysis_pipeline_primes_and_drains_in_order() &&
             automatic_h264_coder_preserves_driver_default() &&
             repeated_input_rotates_away_from_a_lookahead_owned_surface() &&
             teardown_timeout_returns_control_before_a_wedged_destructor() ?
           0 :
           1;
}

#else

TEST(NativeAmfReview, SynchronousReleaseDuringSubmitIsReentrantSafe) {
  EXPECT_TRUE(synchronous_release_during_submit_is_reentrant_safe());
}

TEST(NativeAmfReview, BackpressureRetriesUntilAccepted) {
  EXPECT_TRUE(backpressure_retries_the_same_submission_until_accepted());
}

TEST(NativeAmfReview, RecoveryStateChangesOnlyAfterAcceptance) {
  EXPECT_TRUE(recovery_state_changes_only_after_accepted_input());
}

TEST(NativeAmfReview, PreAnalysisDependentRateControlIsPlannedNatively) {
  EXPECT_TRUE(preanalysis_dependent_rate_control_is_planned_natively());
}

TEST(NativeAmfReview, PreAnalysisPipelinePrimesAndDrainsInOrder) {
  EXPECT_TRUE(preanalysis_pipeline_primes_and_drains_in_order());
}

TEST(NativeAmfReview, AutomaticH264CoderPreservesDriverDefault) {
  EXPECT_TRUE(automatic_h264_coder_preserves_driver_default());
}

TEST(NativeAmfReview, RepeatedInputRotatesAwayFromLookaheadOwnedSurface) {
  EXPECT_TRUE(repeated_input_rotates_away_from_a_lookahead_owned_surface());
}

TEST(NativeAmfReview, TeardownTimeoutReturnsControl) {
  EXPECT_TRUE(teardown_timeout_returns_control_before_a_wedged_destructor());
}

#endif
