import uuid
import random
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from synthetic_data_generator.config import (
    CONFIG, NIGERIAN_CARRIERS, NIGERIAN_STATES, DEVICE_OS, DEVICE_BRANDS
)


@dataclass
class DeviceEvent:
    timestamp: str
    imei: str
    imsi: str
    carrier: str
    device_brand: str
    device_os: str
    os_version: str
    location_state: str
    location_lat: float
    location_lng: float
    is_sim_swap: bool = False
    is_legitimate_upgrade: bool = False
    label: int = 0


@dataclass
class UserDeviceHistory:
    user_id: str
    account_id: str
    events: list[DeviceEvent] = field(default_factory=list)
    has_sim_swap: bool = False
    sim_swap_index: int = -1


# Approximate Nigerian state coordinates
STATE_COORDS = {
    "Lagos":  (6.52, 3.38),  "Abuja":  (9.07, 7.40),
    "Kano":   (12.00, 8.52), "Rivers": (4.82, 7.04),
    "Oyo":    (7.85, 3.93),  "Kaduna": (10.52, 7.44),
    "Anambra":(6.21, 6.94),  "Delta":  (5.89, 5.68),
    "Ogun":   (6.99, 3.47),  "Enugu":  (6.46, 7.55),
}


def _make_imei() -> str:
    return "".join([str(random.randint(0, 9)) for _ in range(15)])


def _make_imsi() -> str:
    return "62" + "".join([str(random.randint(0, 9)) for _ in range(13)])


def _jitter_location(lat: float, lng: float, radius_km: float = 20.0) -> tuple:
    """Add small random displacement to simulate GPS imprecision."""
    lat_offset = random.uniform(-radius_km / 111.0, radius_km / 111.0)
    lng_offset = random.uniform(-radius_km / 111.0, radius_km / 111.0)
    return round(lat + lat_offset, 6), round(lng + lng_offset, 6)


class DeviceProfileGenerator:
    """
    Generates consistent device history per user account.
    Stable IMEI, IMSI, carrier, device model, OS version, and location radius over time.
    Calibrated from GSMA SIM swap signal distributions.
    """

    def __init__(self, seed: int = None):
        seed = seed or CONFIG.sim_swap.random_seed
        random.seed(seed)
        np.random.seed(seed)

    def generate_stable_history(
        self,
        user_id: str,
        n_events: int = None,
        home_state: str = None,
    ) -> UserDeviceHistory:
        n_events = n_events or CONFIG.sim_swap.history_length
        home_state = home_state or random.choice(NIGERIAN_STATES)
        home_lat, home_lng = STATE_COORDS.get(home_state, (6.52, 3.38))

        # Fixed device identity for this user
        imei = _make_imei()
        imsi = _make_imsi()
        carrier = random.choice(NIGERIAN_CARRIERS)
        brand = random.choice(DEVICE_BRANDS)
        os_name = random.choice(DEVICE_OS)
        os_ver = f"{os_name} {random.randint(10, 14)}"

        events = []
        start_time = datetime.utcnow() - timedelta(days=random.randint(60, 365))

        for i in range(n_events):
            event_time = start_time + timedelta(
                hours=i * random.uniform(4, 48)
            )
            lat, lng = _jitter_location(home_lat, home_lng, radius_km=30)

            events.append(DeviceEvent(
                timestamp=event_time.isoformat() + "Z",
                imei=imei,
                imsi=imsi,
                carrier=carrier,
                device_brand=brand,
                device_os=os_name,
                os_version=os_ver,
                location_state=home_state,
                location_lat=lat,
                location_lng=lng,
                is_sim_swap=False,
                is_legitimate_upgrade=False,
                label=0,
            ))

        return UserDeviceHistory(
            user_id=user_id,
            account_id=str(uuid.uuid4())[:8],
            events=events,
            has_sim_swap=False,
        )

    def generate_batch(
        self,
        n_users: int = None,
    ) -> list[UserDeviceHistory]:
        n_users = n_users or CONFIG.sim_swap.n_users
        return [
            self.generate_stable_history(user_id=str(uuid.uuid4())[:8])
            for _ in range(n_users)
        ]

    def events_to_feature_matrix(
        self, history: UserDeviceHistory, feature_dim: int = 32
    ) -> np.ndarray:
        """Convert a device history to a (n_events, feature_dim) numpy array."""
        rows = []
        for ev in history.events:
            features = [
                int(ev.is_sim_swap),
                int(ev.is_legitimate_upgrade),
                ev.location_lat / 90.0,
                ev.location_lng / 180.0,
                hash(ev.imei) % 1000 / 1000.0,
                hash(ev.carrier) % 10 / 10.0,
                hash(ev.device_brand) % 10 / 10.0,
                hash(ev.device_os) % 2 / 2.0,
            ]
            features += [0.0] * (feature_dim - len(features))
            rows.append(features[:feature_dim])
        return np.array(rows, dtype=np.float32)