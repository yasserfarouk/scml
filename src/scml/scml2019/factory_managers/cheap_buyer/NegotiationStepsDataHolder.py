class NegotiationStepsDataHolder:
    def __init__(self):
        self.average_number_of_steps = 0
        self.number_of_negotiations = 0

    def add_new_data(self, number_of_negotiation_steps):
        sum_of_steps = self.average_number_of_steps * self.number_of_negotiations
        sum_of_steps += number_of_negotiation_steps
        self.number_of_negotiations += 1
        self.average_number_of_steps = sum_of_steps / self.number_of_negotiations

    def get_average_number_of_negotiation_steps(self):
        return self.average_number_of_steps

    def get_number_of_negotiations(self):
        return self.number_of_negotiations
