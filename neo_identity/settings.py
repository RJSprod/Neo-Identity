"""The per-generation settings, their choices, and the infotext they write.

The names here are the contract between the accordion, the infotext and the
engine: a radio's choice labels are the values the infotext records, so a pasted
infotext puts the same word back into the same control.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

# --------------------------------------------------------------------------- #
# Choices
# --------------------------------------------------------------------------- #

ALIGNED = "Aligned"
"""The reference's token (h, w) sits on the target's token (h, w): how the
models were trained, and upstream's default."""

BESIDE = "Beside"
"""The reference is moved to the right of the target canvas in RoPE space (and
to the right of any earlier reference placed beside): identity still flows, the
layout pin is gone, so the person can be restaged more freely."""

BAG = "Bag of features"
"""Every token of the reference shares one position, the target's centre, so
attention to it is decided by content alone. Upstream marks this experimental."""

PLACEMENTS = (ALIGNED, BESIDE, BAG)

FIT = "Fit"
"""The v1.2+ geometry: the source is resampled in pixel space to the target
grid's density at a centred, fractional offset; a mismatched aspect ratio is
centre-cropped so the fitted axis lands on the 16 px grid exactly."""

CROP = "Crop"
"""The v1 and v1.1 geometry: centre-crop to the target's aspect ratio, then
resize to the target. Kept for the older weights."""

GEOMETRIES = (FIT, CROP)

GROUNDING_DEFAULT = 768
"""The longest side, in pixels, fed to Qwen3-VL. The released LoRAs trained at
a 768 cap with jitter down to 384; 0 means the image at its own size."""
GROUNDING_MAX = 2048
GROUNDING_STEP = 64

BOOST_DEFAULT = 1.0
"""Neutral: no attention bias at all, and the native kernels untouched."""
BOOST_MIN = 0.25
BOOST_MAX = 8.0
BOOST_STEP = 0.05

# --------------------------------------------------------------------------- #
# Infotext keys
# --------------------------------------------------------------------------- #

INFOTEXT_ENABLED = "Neo Identity"
INFOTEXT_SUBJECT_PLACEMENT = "Neo Identity subject placement"
INFOTEXT_SCENE_PLACEMENT = "Neo Identity scene placement"
INFOTEXT_SUBJECT_BOOST = "Neo Identity subject boost"
INFOTEXT_SCENE_BOOST = "Neo Identity scene boost"
INFOTEXT_GROUNDING = "Neo Identity grounding"
INFOTEXT_GEOMETRY = "Neo Identity geometry"
INFOTEXT_NEGATIVE = "Neo Identity negative grounding"
INFOTEXT_REFERENCES = "Neo Identity references"
"""How many references the recipe used; written for diagnostics, pasted into
nothing, because the images themselves are not in the infotext."""

ENABLED_VALUE = "v2"
"""What ``Neo Identity`` says in the infotext. Anything but ``False`` pastes
back as a ticked accordion, so the value can name the recipe."""


@dataclass(frozen=True)
class Settings:
    subject_placement: str = ALIGNED
    scene_placement: str = ALIGNED
    subject_boost: float = BOOST_DEFAULT
    scene_boost: float = BOOST_DEFAULT
    grounding_px: int = GROUNDING_DEFAULT
    geometry: str = FIT
    ground_negative: bool = True

    @classmethod
    def normalized(cls, **values) -> "Settings":
        """A ``Settings`` from loosely typed UI values, every field held in range.

        A control's value can arrive as a string from the API or an infotext
        paste, as a float where an int is wanted, or as a choice this version
        does not know. Each is coerced or replaced by its default rather than
        refused, so a generation never fails on a control's formatting.
        """
        settings = cls()
        if values.get("subject_placement") in PLACEMENTS:
            settings = replace(settings, subject_placement=values["subject_placement"])
        if values.get("scene_placement") in PLACEMENTS:
            settings = replace(settings, scene_placement=values["scene_placement"])
        if values.get("geometry") in GEOMETRIES:
            settings = replace(settings, geometry=values["geometry"])
        settings = replace(
            settings,
            subject_boost=_boost(values.get("subject_boost")),
            scene_boost=_boost(values.get("scene_boost")),
            grounding_px=_grounding(values.get("grounding_px")),
            ground_negative=_flag(values.get("ground_negative"), True),
        )
        return settings

    def placements(self, count: int) -> list[str]:
        """One placement per reference, in sequence order.

        Upstream's convention, kept: the *last* reference is the subject (the
        only image, or the person in a scene-and-person edit) and every earlier
        one is scene. With three or more references, every reference but the
        first follows the subject's choice.
        """
        if count <= 0:
            return []
        return [self.scene_placement] * (count - 1) + [self.subject_placement]

    def boosts(self, count: int) -> list[float]:
        if count <= 0:
            return []
        return [self.scene_boost] * (count - 1) + [self.subject_boost]

    @property
    def boosted(self) -> bool:
        return self.subject_boost != BOOST_DEFAULT or self.scene_boost != BOOST_DEFAULT

    def infotext(self, references: int | None = None) -> dict:
        """The ``extra_generation_params`` entries for this generation."""
        params = {
            INFOTEXT_ENABLED: ENABLED_VALUE,
            INFOTEXT_SUBJECT_PLACEMENT: self.subject_placement,
            INFOTEXT_SCENE_PLACEMENT: self.scene_placement,
            INFOTEXT_SUBJECT_BOOST: _number(self.subject_boost),
            INFOTEXT_SCENE_BOOST: _number(self.scene_boost),
            INFOTEXT_GROUNDING: str(int(self.grounding_px)),
            INFOTEXT_GEOMETRY: self.geometry,
            INFOTEXT_NEGATIVE: "True" if self.ground_negative else "False",
        }
        if references is not None:
            params[INFOTEXT_REFERENCES] = str(int(references))
        return params

    def describe(self, count: int | None = None) -> str:
        """One line for the console: what the recipe does with ``count``
        references, or with any number when ``count`` is not known yet."""
        tail = f"grounding {int(self.grounding_px) or 'native'}, geometry {self.geometry.lower()}"
        if not self.ground_negative:
            tail += ", negative text-only"
        if count is None:
            parts = [f"subject {self.subject_placement.lower()}", f"scene {self.scene_placement.lower()}"]
            if self.subject_boost != BOOST_DEFAULT:
                parts[0] += f" boost {_number(self.subject_boost)}"
            if self.scene_boost != BOOST_DEFAULT:
                parts[1] += f" boost {_number(self.scene_boost)}"
            return ", ".join(parts) + f"; {tail}"
        if count <= 0:
            return "no reference"
        parts = []
        for index, (placement, boost) in enumerate(zip(self.placements(count), self.boosts(count))):
            role = "subject" if index == count - 1 else "scene"
            text = f"{role} {placement.lower()}"
            if boost != BOOST_DEFAULT:
                text += f" boost {_number(boost)}"
            parts.append(text)
        return ", ".join(parts) + f"; {tail}"


def _boost(value) -> float:
    try:
        boost = float(value)
    except (TypeError, ValueError):
        return BOOST_DEFAULT
    if boost != boost:  # NaN
        return BOOST_DEFAULT
    return min(max(boost, BOOST_MIN), BOOST_MAX)


def _grounding(value) -> int:
    try:
        grounding = int(float(value))
    except (TypeError, ValueError):
        return GROUNDING_DEFAULT
    return min(max(grounding, 0), GROUNDING_MAX)


def _flag(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "no", "off", "")
    return bool(value)


def _number(value: float) -> str:
    """``2.0`` as ``2``, ``1.25`` as ``1.25``: the infotext stays short."""
    text = f"{float(value):g}"
    return text
