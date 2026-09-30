# SPDX-FileCopyrightText: 2026 AnySynth contributors
# SPDX-License-Identifier: Apache-2.0

"""Training semantics, decoder alignment, and gradients through ODE exposure."""

from __future__ import annotations

import math
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from midi_span.training import (
        differentiable_euler,
        field_mse,
        prefix_cross_entropy,
        prefix_logits,
    )


def endpoints(counts):
    """Build legal piano endpoints from per-frame (ON, OFF) counts."""
    value = torch.zeros(1, len(counts), 32, 9)
    value[..., 0] = -1.0
    for frame, bank_counts in enumerate(counts):
        for bank, count in enumerate(bank_counts):
            occupied = value[0, frame, bank * 16:bank * 16 + count]
            occupied[:, 0] = 1.0
            occupied[:, 1] = 1.0
            occupied[:, 3] = 1.0
            occupied[:, 5] = (60.0 - 63.5) / 63.5
            occupied[:, 6] = 2.0 * 80.0 / 127.0 - 1.0
            occupied[:, 7] = 2.0 * math.log1p(80.0) / math.log(128.0) - 1.0
            occupied[:, 8] = -1.0
    return value


@unittest.skipIf(torch is None, "PyTorch is an optional training dependency")
class FieldLossTests(unittest.TestCase):
    def test_piano_fields_and_four_presence_families_have_equal_weight(self):
        clean = endpoints([(1, 1)])
        target = torch.zeros_like(clean)
        prediction = torch.zeros_like(clean)
        prediction[0, 0, 0, 0] = 2.0
        prediction[0, 0, 1:16, 0] = 4.0
        prediction[0, 0, 16, 0] = 6.0
        prediction[0, 0, 17:32, 0] = 8.0
        for lane in (0, 16):
            prediction[0, 0, lane, 1:3] = torch.tensor([1.0, 3.0])
            prediction[0, 0, lane, 3:6] = torch.tensor([1.0, 2.0, 3.0])
            prediction[0, 0, lane, 6:8] = torch.tensor([2.0, 4.0])
            prediction[0, 0, lane, 8] = 3.0
        loss = field_mse(prediction, target, clean)
        for name, expected in {
            "presence": 30.0,
            "program": 5.0,
            "pitch": 14.0 / 3.0,
            "velocity": 10.0,
            "subsample": 9.0,
            "total": 176.0 / 15.0,
        }.items():
            with self.subTest(field=name):
                self.assertAlmostEqual(getattr(loss, name).item(), expected, places=5)

    def test_program_groups_are_balanced_even_with_many_more_piano_notes(self):
        clean = endpoints([(10, 0), (1, 0), (1, 0)])
        clean[0, 1, 0, 1:3] = torch.tensor([-0.5, math.sqrt(3.0) / 2.0])
        clean[0, 2, 0, 1:3] = torch.tensor([-0.5, -math.sqrt(3.0) / 2.0])
        prediction = torch.zeros_like(clean)
        prediction[0, 0, :10, 1:3] = 2.0
        prediction[0, 1, 0, 1:3] = 4.0
        prediction[0, 2, 0, 1:3] = 6.0
        loss = field_mse(prediction, torch.zeros_like(clean), clean)
        self.assertAlmostEqual(loss.program.item(), 56.0 / 3.0, places=5)
        self.assertAlmostEqual(loss.total.item(), 56.0 / 15.0, places=5)

    def test_empty_payload_does_not_affect_loss_or_gradient(self):
        clean = endpoints([(1, 0)])
        prediction = torch.zeros_like(clean)
        prediction[..., 1:] = 99.0
        prediction[0, 0, 0, 1:] = 0.0
        prediction.requires_grad_()
        loss = field_mse(prediction, torch.zeros_like(clean), clean)
        self.assertEqual(loss.total.item(), 0.0)
        loss.total.backward()
        self.assertEqual(torch.count_nonzero(prediction.grad).item(), 0)

    def test_all_empty_canvas_uses_presence_without_diluting_by_absent_fields(self):
        clean = endpoints([(0, 0)])
        prediction = torch.zeros_like(clean)
        prediction[..., :16, 0] = 2.0
        prediction[..., 16:, 0] = 4.0
        prediction[..., 1:] = 100.0
        loss = field_mse(prediction, torch.zeros_like(clean), clean)
        self.assertEqual(loss.presence.item(), 10.0)
        self.assertEqual(loss.total.item(), 10.0)
        for name in ("program", "pitch", "velocity", "subsample"):
            self.assertEqual(getattr(loss, name).item(), 0.0)

    def test_padding_is_excluded_from_values_and_gradients(self):
        clean = endpoints([(1, 1), (16, 16)])
        prediction = torch.ones_like(clean)
        prediction[:, 1] = 100.0
        prediction.requires_grad_()
        mask = torch.tensor([[True, False]])
        loss = field_mse(prediction, torch.zeros_like(clean), clean, valid_mask=mask)
        self.assertEqual(loss.total.item(), 1.0)
        loss.total.backward()
        self.assertEqual(torch.count_nonzero(prediction.grad[:, 1]).item(), 0)
        self.assertGreater(torch.count_nonzero(prediction.grad[:, 0]).item(), 0)

    def test_flat_and_lane_layouts_produce_same_fields(self):
        clean = endpoints([(2, 1), (0, 3)])
        prediction = torch.linspace(-2.0, 2.0, clean.numel()).reshape_as(clean)
        target = clean / 3.0
        lanes = field_mse(prediction, target, clean)
        flat = field_mse(prediction.flatten(2), target.flatten(2), clean.flatten(2))
        for name in ("total", "presence", "program", "pitch", "velocity", "subsample"):
            torch.testing.assert_close(getattr(flat, name), getattr(lanes, name))


@unittest.skipIf(torch is None, "PyTorch is an optional training dependency")
class PrefixLossTests(unittest.TestCase):
    def test_empty_and_full_banks_select_counts_zero_and_sixteen(self):
        clean = endpoints([(0, 16), (16, 0)])
        logits = prefix_logits(clean)
        self.assertEqual(tuple(logits.shape), (1, 2, 2, 17))
        self.assertEqual(logits.argmax(-1).tolist(), [[[0, 16], [16, 0]]])
        self.assertTrue(torch.equal(logits[..., 0], torch.zeros_like(logits[..., 0])))
        loss = prefix_cross_entropy(clean, clean)
        expected = math.log(sum(math.exp(-4.0 * n) for n in range(17))) / math.log(17.0)
        self.assertAlmostEqual(loss.total.item(), expected, places=6)

    def test_uniform_logits_have_normalized_loss_one_with_missing_groups(self):
        clean = endpoints([(0, 0), (0, 0)])
        loss = prefix_cross_entropy(torch.zeros_like(clean), clean)
        self.assertAlmostEqual(loss.total.item(), 1.0, places=6)
        self.assertEqual({name: count.item() for name, count in loss.group_counts.items()}, {
            "on_empty": 2, "on_nonempty": 0,
            "off_empty": 2, "off_nonempty": 0,
        })
        self.assertEqual(loss.group_means["on_nonempty"].item(), 0.0)
        self.assertEqual(loss.group_means["off_nonempty"].item(), 0.0)

    def test_nonempty_frames_are_not_diluted_by_many_empty_frames(self):
        clean = endpoints([(0, 0), (0, 0), (2, 0)])
        prediction = clean.clone()
        prediction[0, 1, :16, 0] = 1.0
        prediction[..., 16:, 0] = 0.0
        logits = prefix_logits(prediction)
        counts = torch.tensor([[[0, 0], [0, 0], [2, 0]]])
        per_bank = (logits.logsumexp(-1) - logits.gather(-1, counts[..., None]).squeeze(-1))
        per_bank = per_bank / math.log(17.0)
        expected = (per_bank[0, :2, 0].mean() + per_bank[0, 2, 0]
                    + per_bank[0, :, 1].mean()) / 3.0
        loss = prefix_cross_entropy(prediction, clean)
        torch.testing.assert_close(loss.total.double(), expected.double())
        self.assertGreater(abs(loss.total.item() - per_bank.mean().item()), 0.1)

    def test_rational_tail_keeps_hard_forward_and_decreasing_nonzero_gradients(self):
        values = torch.tensor([-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0])
        prediction = torch.zeros(len(values), 1, 32, 9)
        prediction[:, 0, 0, 0] = values
        prediction.requires_grad_()
        first_prefix = prefix_logits(prediction)[:, 0, 0, 1]
        torch.testing.assert_close(first_prefix.float(), 4.0 * values.clamp(-1.0, 1.0))
        first_prefix.sum().backward()
        expected = 4.0 / (1.0 + (values.abs() - 1.0).clamp_min(0.0)).square()
        torch.testing.assert_close(prediction.grad[:, 0, 0, 0], expected)
        self.assertEqual(torch.count_nonzero(prediction.grad[..., 1:]).item(), 0)

    def test_padding_and_payload_have_no_prefix_gradient(self):
        clean = endpoints([(2, 1), (16, 16)])
        prediction = torch.zeros_like(clean, requires_grad=True)
        mask = torch.tensor([[True, False]])
        loss = prefix_cross_entropy(prediction, clean, valid_mask=mask)
        loss.total.backward()
        self.assertEqual(torch.count_nonzero(prediction.grad[:, 1]).item(), 0)
        self.assertEqual(torch.count_nonzero(prediction.grad[..., 1:]).item(), 0)
        self.assertGreater(torch.count_nonzero(prediction.grad[:, 0, :, 0]).item(), 0)
        self.assertEqual(sum(count.item() for count in loss.group_counts.values()), 2)

    def test_flat_and_lane_layouts_match(self):
        clean = endpoints([(1, 4), (0, 0)])
        prediction = torch.linspace(-3.0, 3.0, clean.numel()).reshape_as(clean)
        torch.testing.assert_close(prefix_logits(prediction.flatten(2)), prefix_logits(prediction))
        lanes = prefix_cross_entropy(prediction, clean)
        flat = prefix_cross_entropy(prediction.flatten(2), clean.flatten(2))
        torch.testing.assert_close(flat.total, lanes.total)
        for name in lanes.group_counts:
            torch.testing.assert_close(flat.group_means[name], lanes.group_means[name])
            torch.testing.assert_close(flat.group_counts[name], lanes.group_counts[name])


@unittest.skipIf(torch is None, "PyTorch is an optional training dependency")
class EulerExposureTests(unittest.TestCase):
    def test_four_calls_use_left_endpoint_times_and_preserve_layout(self):
        for shape in ((2, 3, 288), (2, 3, 32, 9)):
            with self.subTest(shape=shape):
                times = []

                def velocity(state, time):
                    times.append(time.detach().clone())
                    return torch.full_like(state, 2.0)

                result = differentiable_euler(velocity, torch.ones(shape, dtype=torch.float64))
                self.assertEqual(tuple(result.shape), shape)
                self.assertEqual(result.dtype, torch.float32)
                torch.testing.assert_close(result, torch.full(shape, 3.0))
                self.assertEqual(len(times), 4)
                for time, expected in zip(times, (0.0, 0.25, 0.5, 0.75)):
                    torch.testing.assert_close(time, torch.full((2,), expected))

    def test_model_parameter_and_initial_state_receive_gradients_through_all_steps(self):
        initial = torch.ones(2, 3, 288, requires_grad=True)
        scale = torch.tensor(2.0, requires_grad=True)
        mask = torch.tensor([[True, True, False], [True, False, False]])
        observed = []

        def velocity(state, time):
            observed.append(state.detach().clone())
            return scale * state + (~mask)[..., None].to(state.dtype) * 99.0

        result = differentiable_euler(velocity, initial, valid_mask=mask)
        self.assertEqual(torch.count_nonzero(result[~mask]).item(), 0)
        torch.testing.assert_close(result[mask], torch.full_like(result[mask], 1.5 ** 4))
        result.sum().backward()
        self.assertEqual(torch.count_nonzero(initial.grad[~mask]).item(), 0)
        torch.testing.assert_close(initial.grad[mask], torch.full_like(initial.grad[mask], 1.5 ** 4))
        self.assertAlmostEqual(scale.grad.item(), mask.sum().item() * 288 * 1.5 ** 3, places=4)
        for state in observed:
            self.assertEqual(torch.count_nonzero(state[~mask]).item(), 0)

    def test_invalid_step_count_and_empty_sample_mask_are_rejected(self):
        initial = torch.ones(2, 2, 288)
        velocity = lambda state, time: state
        for steps in (0, -1, 1.5):
            with self.subTest(steps=steps), self.assertRaises((ValueError, TypeError)):
                differentiable_euler(velocity, initial, steps=steps)
        mask = torch.tensor([[True, False], [False, False]])
        with self.assertRaises(ValueError):
            differentiable_euler(velocity, initial, valid_mask=mask)


@unittest.skipIf(torch is None, "PyTorch is an optional training dependency")
class TrainingInputTests(unittest.TestCase):
    def test_mixed_layouts_and_invalid_masks_are_rejected(self):
        clean = endpoints([(1, 1)])
        with self.assertRaises(ValueError):
            field_mse(clean.flatten(2), clean.flatten(2), clean)
        with self.assertRaises(ValueError):
            prefix_cross_entropy(clean.flatten(2), clean)
        for mask in (torch.tensor([[False]]), torch.tensor([[1]]), torch.tensor([True])):
            with self.subTest(mask=mask), self.assertRaises((ValueError, TypeError)):
                field_mse(clean, clean, clean, valid_mask=mask)
            with self.subTest(mask=mask), self.assertRaises((ValueError, TypeError)):
                prefix_cross_entropy(clean, clean, valid_mask=mask)


if __name__ == "__main__":
    unittest.main()
