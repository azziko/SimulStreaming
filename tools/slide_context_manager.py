import glob
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

SLIDE_SEPARATOR = " "


def _load_slides(slides_dir: str) -> List[Tuple[float, str]]:
    """
    Load all *.OSt files from *slides_dir* and return a list of
    (start_seconds, text) tuples sorted chronologically.
    """
    pattern = os.path.join(slides_dir, "*.OSt")
    paths = glob.glob(pattern)
    if not paths:
        logger.warning("No .OSt files found in '%s'", slides_dir)
        return []

    slides: List[Tuple[float, str]] = []
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            timestamp = float(stem)
        except ValueError:
            logger.warning("Skipping file with non-numeric stem: %s", os.path.basename(path))
            continue

        with open(path, "r", encoding="utf-8") as fh:
            text = " ".join(fh.read().split())

        slides.append((timestamp, text))

    slides.sort(key=lambda x: x[0])
    logger.info("Loaded %d slides from '%s'", len(slides), slides_dir)
    return slides

class BaseSlideContextManager(ABC):
    """
    Abstract base class for all slide context strategies.

    Subclasses must implement :meth:`get_slide_text`.
    """

    def __init__(self, slides_dir: str) -> None:
        self.slides: List[Tuple[float, str]] = _load_slides(slides_dir)

    def _slides_up_to(self, current_time: float) -> List[Tuple[float, str]]:
        """Return all slides whose start time is <= *current_time*."""
        return [(t, txt) for t, txt in self.slides if t <= current_time]

    def _current_slide(self, current_time: float) -> Optional[Tuple[float, str]]:
        """Return the most-recently-visible (timestamp, text) pair, or None."""
        active: Optional[Tuple[float, str]] = None
        for start_sec, text in self.slides:
            if start_sec <= current_time:
                active = (start_sec, text)
            else:
                break
        return active

    @abstractmethod
    def get_slide_text(self, current_time: float) -> Optional[str]:
        """
        Return the context string to inject at *current_time* seconds,
        or None if nothing should be injected yet.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(slides={len(self.slides)})"


class SingleSlideContextManager(BaseSlideContextManager):
    """
    Inject **only** the currently-visible slide.

    At any point in time the model sees 
    exactly one slide — the most recent one — with no
    memory of earlier slides.
    """

    def get_slide_text(self, current_time: float) -> Optional[str]:
        entry = self._current_slide(current_time)
        if entry is None:
            return None
        _, text = entry
        logger.debug("[Single] t=%.1fs → 1 slide", current_time)
        return text


class IncrementalSlideContextManager(BaseSlideContextManager):
    """
    Accumulate slides one by one as they become visible.

    Once a slide has been seen it stays in the context forever — each new
    slide is appended to what was already there.  This gives the model
    growing historical awareness of all slides shown so far.
    """

    def get_slide_text(self, current_time: float) -> Optional[str]:
        seen = self._slides_up_to(current_time)
        if not seen:
            return None

        combined = SLIDE_SEPARATOR.join(txt for _, txt in seen)
        logger.debug("[Incremental] t=%.1fs → %d slides accumulated", current_time, len(seen))
        return combined


class AllSlidesContextManager(BaseSlideContextManager):
    """
    Inject **all** slides in chronological order, regardless of the current
    playback position.
    """

    # Pre-compute once; never changes.
    _cached: Optional[str] = None

    def __init__(self, slides_dir: str) -> None:
        super().__init__(slides_dir)
        if self.slides:
            self._cached = SLIDE_SEPARATOR.join(txt for _, txt in self.slides)

    def get_slide_text(self, current_time: float) -> Optional[str]:
        logger.debug("[All] %d slides (static)", len(self.slides))
        return self._cached


class AllSlidesCurrentLastManager(BaseSlideContextManager):
    """
    Inject all slides in chronological order **and** repeat the
    currently-visible slide at the very end of the context string.
    """

    def get_slide_text(self, current_time: float) -> Optional[str]:
        if not self.slides:
            return None

        all_texts  = [txt for _, txt in self.slides]
        current    = self._current_slide(current_time)

        if current is None:
            # No slide visible yet — fall back to all slides without suffix
            logger.debug("[AllCurrentLast] t=%.1fs → no current slide yet", current_time)
            return SLIDE_SEPARATOR.join(all_texts)

        _, current_text = current
        combined = SLIDE_SEPARATOR.join(all_texts) + SLIDE_SEPARATOR + current_text
        logger.debug("[AllCurrentLast] t=%.1fs → all slides + current repeated", current_time)
        return combined

_STRATEGY_MAP = {
    "single":           SingleSlideContextManager,
    "incremental":      IncrementalSlideContextManager,
    "all":              AllSlidesContextManager,
    "all_current_last": AllSlidesCurrentLastManager,
}


class SlideContextManagerFactory:
    """
    Creates slide context managers by strategy name.

    Supported strategies
    --------------------
    "single"           → SingleSlideContextManager
    "incremental"      → IncrementalSlideContextManager
    "all"              → AllSlidesContextManager
    "all_current_last" → AllSlidesCurrentLastManager
    """

    @staticmethod
    def create(strategy: str, slides_dir: str) -> BaseSlideContextManager:
        cls = _STRATEGY_MAP.get(strategy)
        if cls is None:
            raise ValueError(
                f"Unknown slide strategy '{strategy}'. "
                f"Choose from: {list(_STRATEGY_MAP)}"
            )
        return cls(slides_dir)

    @staticmethod
    def choices() -> List[str]:
        """Return the list of valid strategy names"""
        return list(_STRATEGY_MAP)