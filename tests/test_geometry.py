"""The reference geometry, number for number.

The expected values are worked by hand from the trainer's geometry contract
(krea2edit-trainer, docs/GEOMETRY.md) and the node's ``_fit_encode_image``; a
test that only compared two copies of the same code would prove nothing.
"""

import pytest
import torch

from neo_identity import geometry as G
from neo_identity import settings as S


class TestFitPlan:
    def test_a_source_narrower_than_the_target_keeps_its_own_grid(self):
        # 1024 wide, 768 tall, into a 1024 square: the width matches, the height
        # does not, so the reference is 48 token rows centred in 64.
        fit = G.plan_fit(768, 1024, 1024, 1024)
        assert fit.crop == (0, 0, 768, 1024)
        assert fit.output == (768, 1024)
        assert fit.grid == (48, 64)
        assert not fit.fills_target

    def test_an_odd_gap_lands_on_a_half_token_offset(self):
        # The documented case: a 47-row reference in a 64-row target is 8.5
        # tokens from the top, not 8 -- an integer floor is 8 px off centre.
        fit = G.plan_fit(752, 1024, 1024, 1024)
        assert fit.grid == (47, 64)
        assert G.centered_offset(64, 47) == 8.5
        assert G.centered_offset(64, 48) == 8.0
        assert G.centered_offset(64, 64) == 0.0

    def test_a_near_matched_ratio_fills_the_grid_through_a_minimal_crop(self):
        # 1000 tall in a 1024 square is within 8 %: 12 px come off each side of
        # the width and the result is the whole grid.
        fit = G.plan_fit(1000, 1024, 1024, 1024)
        assert fit.crop == (0, 12, 1000, 1000)
        assert fit.output == (1024, 1024)
        assert fit.fills_target

    def test_a_small_source_is_enlarged_to_the_target(self):
        fit = G.plan_fit(512, 512, 1024, 1024)
        assert fit.crop == (0, 0, 512, 512)
        assert fit.output == (1024, 1024)

    def test_a_portrait_source_fits_inside_a_landscape_target(self):
        fit = G.plan_fit(1536, 1024, 768, 1024)
        assert fit.crop == (0, 0, 1536, 1024)
        assert fit.output == (768, 512)
        assert fit.grid == (48, 32)
        assert G.centered_offset(64, 32) == 16.0

    def test_the_source_is_cropped_to_the_grid_rather_than_squashed(self):
        # 753 tall: the fitted height snaps to 752, and the source is cropped
        # to 752 rows before the resize so nothing is squashed by a pixel.
        fit = G.plan_fit(753, 1024, 1024, 1024)
        assert fit.crop == (0, 0, 752, 1024)
        assert fit.output == (752, 1024)

    def test_a_genuinely_mismatched_scale_crops_to_the_grid_at_scale(self):
        # 2000x1500 into 1024x1024: scale 0.512, fitted height 1024 -> exact,
        # fitted width 768 after the /16 floor; the crop keeps the scale exact.
        fit = G.plan_fit(2000, 1500, 1024, 1024)
        assert fit.output == (1024, 768)
        y0, x0, ch, cw = fit.crop
        assert (ch, cw) == (2000, 1500)
        assert (y0, x0) == (0, 0)

    def test_the_crop_geometry_centre_crops_to_the_ratio_then_fills(self):
        fit = G.plan_crop(768, 1024, 1024, 1024)
        assert fit.crop == (0, 128, 768, 768)
        assert fit.output == (1024, 1024)
        assert fit.geometry == S.CROP

    def test_plan_chooses_by_geometry_name(self):
        assert G.plan(768, 1024, 1024, 1024, S.FIT).geometry == S.FIT
        assert G.plan(768, 1024, 1024, 1024, S.CROP).geometry == S.CROP

    def test_a_zero_size_is_refused(self):
        with pytest.raises(ValueError):
            G.plan_fit(0, 1024, 1024, 1024)

    def test_describe_names_the_sizes(self):
        text = G.plan_fit(768, 1024, 1024, 1024).describe()
        assert "1024x768 -> 1024x768" in text and "grid 64x48" in text


class TestApplyPlan:
    def test_the_output_has_the_planned_size_and_stays_in_range(self):
        pixels = torch.rand(1, 3, 753, 1024) * 1.5  # overshoot on purpose
        fitted, fit = G.fit_pixels(pixels, 1024, 1024)
        assert tuple(fitted.shape) == (1, 3, 752, 1024)
        assert fitted.min() >= 0 and fitted.max() <= 1
        assert fit.output == (752, 1024)

    def test_a_crop_without_a_resize_is_a_plain_window(self):
        pixels = torch.rand(1, 3, 1000, 1024)
        fit = G.FitPlan((1000, 1024), (1000, 1000), (0, 12, 1000, 1000), (1000, 1000), S.FIT)
        out = G.apply_plan(pixels, fit)
        assert torch.equal(out, pixels[..., :, 12:1012])

    def test_a_plan_for_another_source_is_refused(self):
        fit = G.plan_fit(768, 1024, 1024, 1024)
        with pytest.raises(ValueError):
            G.apply_plan(torch.rand(1, 3, 700, 1024), fit)


class TestGrounding:
    def test_the_longest_side_is_capped(self):
        out = G.grounding_pixels(torch.rand(1, 3, 1536, 1024), 768)
        assert tuple(out.shape[-2:]) == (768, 512)

    def test_a_small_image_is_not_enlarged(self):
        pixels = torch.rand(1, 3, 500, 400)
        assert G.grounding_pixels(pixels, 768) is pixels

    def test_zero_means_the_image_as_it_is(self):
        pixels = torch.rand(1, 3, 1536, 1024)
        assert G.grounding_pixels(pixels, 0) is pixels


def _place_refs_upstream(ref_ids, modes, h_, w_):
    """Upstream's ``_place_refs`` (lbouaraba/comfyui-krea2edit v2.0.0,
    Apache-2.0), kept here as the reference the port is held to."""
    if isinstance(modes, str):
        modes = [modes] * len(ref_ids)
    out, x0 = [], float(w_)
    for ids, mode in zip(ref_ids, modes):
        if mode.startswith("aligned"):
            out.append(ids)
            continue
        ids = ids.clone()
        if mode.startswith("beside"):
            ids[..., 2] = ids[..., 2] - ids[..., 2].min() + x0
            x0 = float(ids[..., 2].max()) + 1.0
        elif mode.startswith("bag"):
            ids[..., 1] = (h_ - 1) / 2.0
            ids[..., 2] = (w_ - 1) / 2.0
        out.append(ids)
    return out


UPSTREAM_NAMES = {S.ALIGNED: "aligned (trained)", S.BESIDE: "beside (right of target)", S.BAG: "bag of features"}


class TestPositionIds:
    def test_the_target_is_frame_zero_at_stride_one(self):
        ids = G.target_ids(2, 3, 4, "cpu")
        assert tuple(ids.shape) == (2, 12, 3)
        assert torch.all(ids[..., 0] == 0)
        assert ids[0, 5].tolist() == [0.0, 1.0, 1.0]

    def test_a_reference_sits_at_its_centred_fractional_offset(self):
        ids = G.reference_ids(1, 2, 47, 64, 64, 64, "cpu")
        assert torch.all(ids[..., 0] == 2)
        assert ids[0, 0].tolist() == [2.0, 8.5, 0.0]
        assert ids[0, -1].tolist() == [2.0, 8.5 + 46, 63.0]

    def test_frame_zero_is_not_a_reference(self):
        with pytest.raises(ValueError):
            G.reference_ids(1, 0, 4, 4, 8, 8, "cpu")

    @pytest.mark.parametrize(
        "placements",
        [
            (S.ALIGNED, S.ALIGNED),
            (S.ALIGNED, S.BESIDE),
            (S.BESIDE, S.BESIDE),
            (S.BAG, S.ALIGNED),
            (S.BESIDE, S.BAG, S.BESIDE),
        ],
    )
    def test_placement_matches_upstream(self, placements):
        h_, w_ = 20, 30
        grids = [(20, 30), (12, 18), (8, 8)][: len(placements)]
        ids = [G.reference_ids(1, i + 1, gh, gw, h_, w_, "cpu") for i, (gh, gw) in enumerate(grids)]
        ours = G.place(ids, list(placements), h_, w_)
        theirs = _place_refs_upstream(ids, [UPSTREAM_NAMES[p] for p in placements], h_, w_)
        for a, b in zip(ours, theirs):
            assert torch.equal(a, b)

    def test_two_beside_references_do_not_share_a_column(self):
        h_, w_ = 16, 16
        ids = [G.reference_ids(1, 1, 16, 16, h_, w_, "cpu"), G.reference_ids(1, 2, 8, 8, h_, w_, "cpu")]
        first, second = G.place(ids, [S.BESIDE, S.BESIDE], h_, w_)
        assert first[..., 2].min() == 16 and first[..., 2].max() == 31
        assert second[..., 2].min() == 32 and second[..., 2].max() == 39
        assert torch.equal(second[..., 1], ids[1][..., 1])  # vertical placement unchanged

    def test_bag_puts_every_token_at_the_centre(self):
        ids = G.place([G.reference_ids(1, 1, 4, 6, 9, 11, "cpu")], [S.BAG], 9, 11)[0]
        assert torch.all(ids[..., 1] == 4.0) and torch.all(ids[..., 2] == 5.0)
        assert torch.all(ids[..., 0] == 1)

    def test_a_placement_per_reference_is_required(self):
        with pytest.raises(ValueError):
            G.place([G.target_ids(1, 2, 2, "cpu")], [], 2, 2)

    def test_an_unknown_placement_is_refused(self):
        with pytest.raises(ValueError):
            G.place([G.reference_ids(1, 1, 2, 2, 4, 4, "cpu")], ["sideways"], 4, 4)


class TestGrids:
    def test_the_padded_grid_rounds_up_to_the_patch(self):
        assert G.padded_grid(64, 64, 2) == (32, 32)
        assert G.padded_grid(65, 64, 2) == (33, 32)
