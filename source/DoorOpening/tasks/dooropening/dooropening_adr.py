import copy


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
        for term_name, term_params in self.adr_cfg_dict.items():
            if term_name == "num_increments":
                continue
            term = self.event_manager.get_term_cfg(term_name)
            for param_name, param_values in term_params.items():
                lower_limit_inc = (
                    self.adr_cfg_dict[term_name][param_name][0] - self.adr_cfg_dict_initial[term_name][param_name][0]
                ) / float(self.adr_cfg_dict["num_increments"])
                lower_limit = lower_limit_inc * self.increment_counter + self.adr_cfg_dict_initial[term_name][param_name][0]

                upper_limit_inc = (
                    self.adr_cfg_dict[term_name][param_name][1] - self.adr_cfg_dict_initial[term_name][param_name][1]
                ) / float(self.adr_cfg_dict["num_increments"])
                upper_limit = upper_limit_inc * self.increment_counter + self.adr_cfg_dict_initial[term_name][param_name][1]

                term.params[param_name] = (lower_limit, upper_limit)

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
        param_slope = (upper_limit - lower_limit) / float(self.adr_cfg_dict["num_increments"])
        return param_slope * self.increment_counter + lower_limit
