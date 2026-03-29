import copy
import math


class DoorOpeningADR:
    """Tracks the current ADR stage for reset-time EventTerms and env-side custom noise."""

    def __init__(self, event_manager, adr_cfg_dict, adr_custom_cfg_dict):
        self.event_manager = event_manager
        self.adr_cfg_dict = adr_cfg_dict
        self.adr_custom_cfg_dict = adr_custom_cfg_dict

        self.adr_cfg_dict_initial = copy.deepcopy(adr_cfg_dict)
        self.save_param_ranges()

        self.increment_counter = 0

    def save_param_ranges(self):
        # Snapshot the nominal EventTerm ranges so ADR can widen from the base task instead of compounding updates.
        for term_name, term_params in self.adr_cfg_dict.items():
            if term_name == "num_increments":
                continue
            term = self.event_manager.get_term_cfg(term_name)
            for param_name in term_params:
                self.adr_cfg_dict_initial[term_name][param_name] = copy.deepcopy(term.params[param_name])

    def increase_ranges(self, increase_counter: bool = True):
        if self.increment_counter >= self.adr_cfg_dict["num_increments"]:
            self.increment_counter = self.adr_cfg_dict["num_increments"]
        elif increase_counter:
            self.increment_counter += 1

        # EventTerms store their active ranges in-place, so ADR just interpolates each range endpoint for the current stage.
        fraction = self.get_increment_fraction()
        for term_name, term_params in self.adr_cfg_dict.items():
            if term_name == "num_increments":
                continue
            term = self.event_manager.get_term_cfg(term_name)
            for param_name, param_values in term_params.items():
                target_range, interpolation = self._parse_range_cfg(param_values)
                term.params[param_name] = self.interpolate_range(
                    self.adr_cfg_dict_initial[term_name][param_name],
                    target_range,
                    fraction,
                    interpolation,
                )

    def set_num_increments(self, num_increments: int):
        self.increment_counter = num_increments
        self.increase_ranges(increase_counter=False)

    def get_increment_fraction(self) -> float:
        num_increments = float(self.adr_cfg_dict["num_increments"])
        if num_increments <= 0:
            return 0.0
        return float(self.increment_counter) / num_increments

    def get_term_param_range(self, term_name: str, param_name: str):
        return self.event_manager.get_term_cfg(term_name).params[param_name]

    def get_custom_param_value(self, param_group: str, param_name: str):
        upper_limit = self.adr_custom_cfg_dict[param_group][param_name][1]
        lower_limit = self.adr_custom_cfg_dict[param_group][param_name][0]

        # Reset noise, observation noise, and target noise are sampled in the env, so ADR exposes them as scalars.
        return self.interpolate_scalar(lower_limit, upper_limit, self.get_increment_fraction())

    def get_interpolated_range(
        self,
        initial_range: tuple[float, float],
        target_range: tuple[float, float],
        interpolation: str = "linear",
    ) -> tuple[float, float]:
        return self.interpolate_range(initial_range, target_range, self.get_increment_fraction(), interpolation)

    @staticmethod
    def _parse_range_cfg(param_cfg) -> tuple[tuple[float, float], str]:
        if isinstance(param_cfg, dict):
            return tuple(param_cfg["range"]), param_cfg.get("interpolation", "linear")
        return tuple(param_cfg), "linear"

    @staticmethod
    def interpolate_scalar(start: float, end: float, fraction: float) -> float:
        fraction = max(0.0, min(1.0, float(fraction)))
        return (1.0 - fraction) * float(start) + fraction * float(end)

    @classmethod
    def interpolate_range(
        cls,
        initial_range: tuple[float, float],
        target_range: tuple[float, float],
        fraction: float,
        interpolation: str = "linear",
    ) -> tuple[float, float]:
        lower_initial, upper_initial = (float(initial_range[0]), float(initial_range[1]))
        lower_target, upper_target = (float(target_range[0]), float(target_range[1]))
        fraction = max(0.0, min(1.0, float(fraction)))

        if interpolation == "linear":
            return (
                cls.interpolate_scalar(lower_initial, lower_target, fraction),
                cls.interpolate_scalar(upper_initial, upper_target, fraction),
            )
        if interpolation == "log":
            positive_values = (lower_initial, upper_initial, lower_target, upper_target)
            if any(value <= 0.0 for value in positive_values):
                raise ValueError("Log interpolation requires strictly positive range bounds.")
            return (
                math.exp(cls.interpolate_scalar(math.log(lower_initial), math.log(lower_target), fraction)),
                math.exp(cls.interpolate_scalar(math.log(upper_initial), math.log(upper_target), fraction)),
            )
        raise ValueError(f"Unsupported interpolation mode: {interpolation}")
