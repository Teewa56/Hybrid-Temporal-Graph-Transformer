import random
import numpy as np
from copy import deepcopy
from datetime import datetime, timedelta

from synthetic_data_generator.config import CONFIG, NIGERIAN_CARRIERS, NIGERIAN_STATES
from synthetic_data_generator.sim_swap.device_profile_generator import (
    DeviceProfileGenerator,
    DeviceEvent,
    UserDeviceHistory,
    _make_imei,
    _make_imsi,
    _jitter_location,
    STATE_COORDS,
)


class HandoverEventSimulator:
    """
    Injects SIM Swap Handover Events into stable device histories.

    A Handover Event = abrupt IMEI + IMSI + carrier change
    followed within a short window by a high-value transaction attempt.

    Also simulates legitimate device upgrades (gradual onboarding pattern)
    so the model learns the difference between fraud and a real phone change.
    """

    def __init__(self, seed: int = None):
        seed = seed or CONFIG.sim_swap.random_seed
        random.seed(seed)
        np.random.seed(seed)
        self.profile_gen = DeviceProfileGenerator(seed=seed)

    def inject_sim_swap(self, history: UserDeviceHistory) -> UserDeviceHistory:
        """Inject a malicious SIM swap at a random point in the history."""
        h = deepcopy(history)
        if len(h.events) < 5:
            return h

        swap_idx = random.randint(len(h.events) // 2, len(h.events) - 1)
        swap_time = datetime.fromisoformat(
            h.events[swap_idx].timestamp.replace("Z", "")
        )

        # New device identity — completely different from history
        new_imei = _make_imei()
        new_imsi = _make_imsi()
        new_carrier = random.choice([c for c in NIGERIAN_CARRIERS if c != h.events[0].carrier])
        foreign_state = random.choice(
            [s for s in NIGERIAN_STATES if s != h.events[0].location_state]
        )
        f_lat, f_lng = STATE_COORDS.get(foreign_state, (9.07, 7.40))
        f_lat, f_lng = _jitter_location(f_lat, f_lng, radius_km=5)

        # Inject the swap event
        swap_event = DeviceEvent(
            timestamp=swap_time.isoformat() + "Z",
            imei=new_imei,
            imsi=new_imsi,
            carrier=new_carrier,
            device_brand=random.choice(["Samsung", "Tecno", "Infinix"]),
            device_os="Android",
            os_version="Android 12",
            location_state=foreign_state,
            location_lat=f_lat,
            location_lng=f_lng,
            is_sim_swap=True,
            is_legitimate_upgrade=False,
            label=1,
        )
        h.events.insert(swap_idx, swap_event)

        # Label post-swap events within the window as fraudulent
        post_swap_window = CONFIG.sim_swap.post_swap_window
        for i in range(swap_idx + 1, min(swap_idx + 1 + post_swap_window, len(h.events))):
            h.events[i].imei = new_imei
            h.events[i].imsi = new_imsi
            h.events[i].carrier = new_carrier
            h.events[i].location_state = foreign_state
            h.events[i].label = 1

        h.has_sim_swap = True
        h.sim_swap_index = swap_idx
        return h

    def inject_legitimate_upgrade(self, history: UserDeviceHistory) -> UserDeviceHistory:
        """
        Simulate a real phone upgrade — gradual onboarding pattern.
        The new device appears first with small transactions,
        and the transition spans several days.
        """
        h = deepcopy(history)
        if len(h.events) < 5:
            return h

        upgrade_idx = random.randint(len(h.events) // 2, len(h.events) - 1)
        upgrade_time = datetime.fromisoformat(
            h.events[upgrade_idx].timestamp.replace("Z", "")
        )

        new_imei = _make_imei()
        new_imsi = _make_imsi()
        # Same carrier — people rarely change carriers on upgrade
        same_carrier = h.events[0].carrier
        same_state = h.events[0].location_state
        lat, lng = STATE_COORDS.get(same_state, (6.52, 3.38))

        # Gradual transition: overlap period where both devices appear
        for i, days_offset in enumerate([0, 1, 3, 7]):
            t = upgrade_time + timedelta(days=days_offset)
            event = DeviceEvent(
                timestamp=t.isoformat() + "Z",
                imei=new_imei,
                imsi=new_imsi,
                carrier=same_carrier,
                device_brand=random.choice(["Samsung", "iPhone"]),
                device_os=random.choice(["Android", "iOS"]),
                os_version="Latest",
                location_state=same_state,
                location_lat=lat + random.uniform(-0.01, 0.01),
                location_lng=lng + random.uniform(-0.01, 0.01),
                is_sim_swap=False,
                is_legitimate_upgrade=True,
                label=0,
            )
            h.events.insert(upgrade_idx + i, event)

        return h

    def simulate_batch(
        self,
        histories: list[UserDeviceHistory],
    ) -> list[UserDeviceHistory]:
        """
        Inject SIM swaps and legitimate upgrades into a batch of histories.
        Returns all histories with fraud labels applied.
        """
        n_swaps = CONFIG.sim_swap.n_sim_swap_events
        upgrade_rate = CONFIG.sim_swap.legitimate_upgrade_rate

        result = []
        swap_count = 0

        for history in histories:
            if swap_count < n_swaps and random.random() < (n_swaps / len(histories)):
                result.append(self.inject_sim_swap(history))
                swap_count += 1
            elif random.random() < upgrade_rate:
                result.append(self.inject_legitimate_upgrade(history))
            else:
                result.append(history)

        print(f"📱 SIM Swap simulation: {swap_count} swaps injected "
              f"across {len(histories)} users.")
        return result