"""The settings: coercion from UI values, the per-reference roles, the infotext."""

from neo_identity import settings as S


class TestNormalized:
    def test_defaults(self):
        s = S.Settings.normalized()
        assert s == S.Settings()
        assert s.subject_placement == S.ALIGNED and s.grounding_px == 768 and s.ground_negative

    def test_strings_from_the_api_or_a_paste_are_coerced(self):
        s = S.Settings.normalized(subject_placement="Beside", scene_placement="Bag of features", subject_boost="2", scene_boost=1.5, grounding_px="512.0", geometry="Crop", ground_negative="False")
        assert s.subject_placement == S.BESIDE and s.scene_placement == S.BAG
        assert s.subject_boost == 2.0 and s.scene_boost == 1.5
        assert s.grounding_px == 512 and s.geometry == S.CROP and s.ground_negative is False

    def test_unknown_choices_fall_back_to_the_defaults(self):
        s = S.Settings.normalized(subject_placement="sideways", geometry="stretch", subject_boost="lots", grounding_px=None)
        assert s.subject_placement == S.ALIGNED and s.geometry == S.FIT
        assert s.subject_boost == S.BOOST_DEFAULT and s.grounding_px == S.GROUNDING_DEFAULT

    def test_values_are_held_in_range(self):
        s = S.Settings.normalized(subject_boost=100, scene_boost=0, grounding_px=99999)
        assert s.subject_boost == S.BOOST_MAX and s.scene_boost == S.BOOST_MIN
        assert s.grounding_px == S.GROUNDING_MAX
        assert S.Settings.normalized(grounding_px=-5).grounding_px == 0

    def test_the_negative_flag_reads_the_usual_words(self):
        assert S.Settings.normalized(ground_negative="off").ground_negative is False
        assert S.Settings.normalized(ground_negative="True").ground_negative is True
        assert S.Settings.normalized(ground_negative=0).ground_negative is False


class TestRoles:
    def test_the_last_reference_is_the_subject(self):
        s = S.Settings(subject_placement=S.BESIDE, scene_placement=S.ALIGNED, subject_boost=2.0, scene_boost=1.5)
        assert s.placements(1) == [S.BESIDE]
        assert s.placements(2) == [S.ALIGNED, S.BESIDE]
        assert s.placements(3) == [S.ALIGNED, S.ALIGNED, S.BESIDE]
        assert s.boosts(2) == [1.5, 2.0]
        assert s.placements(0) == [] and s.boosts(0) == []

    def test_boosted_means_either_dial_moved(self):
        assert not S.Settings().boosted
        assert S.Settings(scene_boost=1.2).boosted


class TestInfotext:
    def test_every_setting_is_written_and_reads_back(self):
        s = S.Settings(subject_placement=S.BESIDE, subject_boost=2.0, grounding_px=512, geometry=S.CROP, ground_negative=False)
        params = s.infotext(references=2)
        assert params[S.INFOTEXT_ENABLED] == "v2"
        assert params[S.INFOTEXT_SUBJECT_PLACEMENT] == "Beside"
        assert params[S.INFOTEXT_SUBJECT_BOOST] == "2" and params[S.INFOTEXT_SCENE_BOOST] == "1"
        assert params[S.INFOTEXT_GROUNDING] == "512" and params[S.INFOTEXT_GEOMETRY] == "Crop"
        assert params[S.INFOTEXT_NEGATIVE] == "False" and params[S.INFOTEXT_REFERENCES] == "2"
        again = S.Settings.normalized(
            subject_placement=params[S.INFOTEXT_SUBJECT_PLACEMENT],
            scene_placement=params[S.INFOTEXT_SCENE_PLACEMENT],
            subject_boost=params[S.INFOTEXT_SUBJECT_BOOST],
            scene_boost=params[S.INFOTEXT_SCENE_BOOST],
            grounding_px=params[S.INFOTEXT_GROUNDING],
            geometry=params[S.INFOTEXT_GEOMETRY],
            ground_negative=params[S.INFOTEXT_NEGATIVE],
        )
        assert again == s

    def test_the_reference_count_is_optional(self):
        assert S.INFOTEXT_REFERENCES not in S.Settings().infotext()

    def test_describe_names_the_roles(self):
        s = S.Settings(subject_placement=S.BESIDE, subject_boost=2.0)
        assert s.describe(2) == "scene aligned, subject beside boost 2; grounding 768, geometry fit"
        assert s.describe(0) == "no reference"
        assert "subject beside boost 2" in s.describe() and "scene aligned" in s.describe()
        assert S.Settings(grounding_px=0, ground_negative=False).describe(1).endswith("grounding native, geometry fit, negative text-only")
