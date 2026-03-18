"""
Spotify API integration module and legacy /api/spotify endpoint.

This file serves two purposes:
1. Shared Spotify service helpers used by orchestrator.py.
2. Backward-compatible Flask entrypoint for deployments and READMEs that
   still point at /api/spotify.
"""

from __future__ import annotations

import random
import threading
from base64 import b64encode
from dataclasses import dataclass
from typing import Any, Optional

import requests
from flask import Flask, Response, request

from .config import spotify_config, svg_config, validate_background_type, validate_hex_color
from .exceptions import APIError, AuthenticationError, MusicWidgetError, NoTracksError


app = Flask(__name__)


@dataclass
class TrackInfo:
    """Normalized track information."""

    is_playing: bool
    track_name: str
    artist_name: str
    album_name: str
    album_art_url: str
    track_url: str
    artist_url: str
    track_id: str = ""


@dataclass
class AudioFeatures:
    """Audio features for a track from Spotify's audio analysis."""

    tempo: float
    energy: float
    danceability: float
    valence: float
    loudness: float

    @property
    def beat_duration_ms(self) -> int:
        """Calculate duration of one beat in milliseconds."""
        if self.tempo <= 0:
            return 500
        return int(60000 / self.tempo)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for template use."""
        return {
            "tempo": self.tempo,
            "energy": self.energy,
            "danceability": self.danceability,
            "valence": self.valence,
            "loudness": self.loudness,
            "beat_duration_ms": self.beat_duration_ms,
        }


class SpotifyTokenManager:
    """Thread-safe token manager for Spotify API authentication."""

    def __init__(self) -> None:
        self._token: Optional[str] = None
        self._lock = threading.Lock()

    def _get_auth_header(self) -> str:
        """Get base64 encoded authorization header."""
        credentials = f"{spotify_config.client_id}:{spotify_config.client_secret}"
        return b64encode(credentials.encode()).decode("ascii")

    def _refresh_token(self) -> str:
        """Refresh the Spotify access token."""
        data = {
            "grant_type": "refresh_token",
            "refresh_token": spotify_config.refresh_token,
        }
        headers = {"Authorization": f"Basic {self._get_auth_header()}"}

        try:
            response = requests.post(
                spotify_config.token_url,
                data=data,
                headers=headers,
                timeout=10,
                verify=False,
            )
            response.raise_for_status()
            result = response.json()

            if "access_token" not in result:
                raise AuthenticationError("Spotify", "No access token in response")

            return result["access_token"]
        except requests.RequestException as e:
            raise AuthenticationError("Spotify", str(e)) from e

    def get_token(self, force_refresh: bool = False) -> str:
        """Get a valid access token, refreshing if necessary."""
        with self._lock:
            if self._token is None or force_refresh:
                self._token = self._refresh_token()
            return self._token

    def invalidate(self) -> None:
        """Invalidate the current token, forcing refresh on next use."""
        with self._lock:
            self._token = None


_token_manager = SpotifyTokenManager()


def is_configured() -> bool:
    """Check if Spotify environment variables are properly configured."""
    return spotify_config.is_configured()


def _api_get(url: str, retry_on_auth_error: bool = True) -> dict[str, Any]:
    """Make an authenticated GET request to the Spotify API."""
    token = _token_manager.get_token()
    headers = {"Authorization": f"Bearer {token}"}

    try:
        response = requests.get(url, headers=headers, timeout=10, verify=False)

        if response.status_code == 401 and retry_on_auth_error:
            _token_manager.invalidate()
            return _api_get(url, retry_on_auth_error=False)

        if response.status_code == 204:
            raise NoTracksError("Spotify")

        if not response.ok:
            raise APIError("Spotify", response.status_code, response.text)

        return response.json()
    except requests.RequestException as e:
        raise APIError("Spotify", 0, str(e)) from e


def get_recent_tracks(limit: int = 10) -> dict[str, Any]:
    """Fetch recent tracks from Spotify."""
    limit = min(max(1, limit), 50)
    url = f"{spotify_config.recently_played_url}?limit={limit}"
    return _api_get(url)


def get_audio_features(track_id: str) -> Optional[AudioFeatures]:
    """Fetch audio features for a track from Spotify."""
    if not track_id:
        return None

    try:
        url = f"https://api.spotify.com/v1/audio-features/{track_id}"
        data = _api_get(url)

        if not data:
            return None

        return AudioFeatures(
            tempo=data.get("tempo", 120.0),
            energy=data.get("energy", 0.5),
            danceability=data.get("danceability", 0.5),
            valence=data.get("valence", 0.5),
            loudness=data.get("loudness", -10.0),
        )
    except (APIError, NoTracksError):
        return None


def _can_fallback_to_recent_tracks(error: APIError) -> bool:
    """
    Detect Spotify player API failures where recently played can still work.

    Spotify's currently-playing endpoint can return HTTP 403 when the account
    tied to the app is not eligible for player-state access. In that case we
    still want to render the widget using recently played tracks instead of
    surfacing an error card to GitHub.
    """
    message = error.message.lower()
    return "http 403" in message and "premium subscription required" in message


def _extract_track_info(item: dict[str, Any], is_playing: bool) -> TrackInfo:
    """Extract normalized track information from a Spotify item."""
    album_art_url = ""
    images = item.get("album", {}).get("images", [])
    if images:
        album_art_url = images[1]["url"] if len(images) > 1 else images[0]["url"]

    artists = item.get("artists", [{}])
    first_artist = artists[0] if artists else {}

    track_id = item.get("id", "")
    if not track_id:
        uri = item.get("uri", "")
        if uri.startswith("spotify:track:"):
            track_id = uri.split(":")[-1]

    return TrackInfo(
        is_playing=is_playing,
        track_name=item.get("name", "Unknown Track"),
        artist_name=first_artist.get("name", "Unknown Artist"),
        album_name=item.get("album", {}).get("name", "Unknown Album"),
        album_art_url=album_art_url,
        track_url=item.get("external_urls", {}).get("spotify", ""),
        artist_url=first_artist.get("external_urls", {}).get("spotify", ""),
        track_id=track_id,
    )


def get_now_playing() -> dict[str, Any]:
    """Get the currently playing or most recently played track from Spotify."""
    is_playing = False
    item: Optional[dict[str, Any]] = None

    try:
        data = _api_get(spotify_config.now_playing_url)
        if data and "item" in data:
            is_playing = data.get("is_playing", False)
            item = data["item"]
    except NoTracksError:
        pass
    except APIError as e:
        if not _can_fallback_to_recent_tracks(e):
            raise

    if item is None:
        data = _api_get(f"{spotify_config.recently_played_url}?limit=10")
        items = data.get("items", [])

        if not items:
            raise NoTracksError("Spotify")

        item = items[random.randint(0, len(items) - 1)]["track"]
        is_playing = False

    track_info = _extract_track_info(item, is_playing)
    audio_features = get_audio_features(track_info.track_id)

    return {
        "is_playing": track_info.is_playing,
        "track_name": track_info.track_name,
        "artist_name": track_info.artist_name,
        "album_name": track_info.album_name,
        "album_art_url": track_info.album_art_url,
        "track_url": track_info.track_url,
        "artist_url": track_info.artist_url,
        "audio_features": audio_features.to_dict() if audio_features else None,
    }


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def catch_all(path: str) -> Response:
    """
    Backward-compatible image endpoint for deployments that still use /api/spotify.

    This reuses the maintained SVG renderer from orchestrator.py so legacy README
    URLs keep working after newer refactors.
    """
    from .orchestrator import make_error_svg, make_svg

    raw_background = request.args.get("background_color", "")
    raw_border = request.args.get("border_color", "")
    raw_bg_type = request.args.get("background_type", "")

    background_color = validate_hex_color(raw_background, svg_config.default_background)
    border_color = validate_hex_color(raw_border, svg_config.default_border)
    background_type = validate_background_type(raw_bg_type, svg_config.default_background_type)
    show_status = request.args.get("show_status", "").lower() in ("true", "1", "yes")
    is_compact = request.args.get("compact", "").lower() in ("true", "1", "yes")

    try:
        track_data = get_now_playing()
    except MusicWidgetError as e:
        return make_error_svg(e.message, e.status_code)
    except Exception as e:
        return make_error_svg(f"Error: {str(e)}", 500)

    svg = make_svg(track_data, background_color, border_color, background_type, show_status, is_compact)
    resp = Response(svg, mimetype="image/svg+xml")
    resp.headers["Cache-Control"] = "s-maxage=1"
    return resp


if __name__ == "__main__":
    app.run(debug=True)
