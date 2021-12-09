import pandas


TOTAL_POP = 10000000
MONTHLY_SUPPLY = 1000000
ONE_DOSE_EFFICACY = 0.85
TWO_DOSE_EFFICACY = 0.95


def worried():
    results = list()
    # print("Worried:")
    no_doses, one_dose, two_doses = TOTAL_POP, 0, 0
    total_supply, held_over = 0, 0
    for i in range(1000):
        new_supply = MONTHLY_SUPPLY if total_supply < (TOTAL_POP * 2) else 0

        if not (new_supply or held_over):
            break

        total_supply += new_supply

        # Vaccinate those held over with a second dose
        if held_over > 0:
            two_doses += held_over
            one_dose -= held_over
            held_over = 0

        # Use half of the new supply to vaccinate people with first dose
        half_new = new_supply / 2
        no_doses -= half_new
        one_dose += half_new
        held_over += half_new

        herd_immunity = ((one_dose * ONE_DOSE_EFFICACY) + (two_doses * TWO_DOSE_EFFICACY)) / TOTAL_POP
        herd_immunity = int((herd_immunity * 100) + 0.5)
        results.append(f'{herd_immunity}%')

    return results


def confident():
    results = list()
    # print("Confident:")
    no_doses, one_dose, two_doses = TOTAL_POP, 0, 0
    total_supply = 0
    for i in range(1000):
        new_supply = MONTHLY_SUPPLY if total_supply < (TOTAL_POP * 2) else 0

        if not new_supply:
            break

        total_supply += new_supply

        # Give second dose to as many as you can who have had first dose
        if one_dose > 0:
            shots_given = min(one_dose, new_supply)
            two_doses += shots_given
            one_dose -= shots_given
            new_supply -= shots_given

        # Use what you have left to give first doses
        no_doses -= new_supply
        one_dose += new_supply

        herd_immunity = ((one_dose * ONE_DOSE_EFFICACY) + (two_doses * TWO_DOSE_EFFICACY)) / TOTAL_POP
        herd_immunity = int((herd_immunity * 100) + 0.5)
        results.append(f'{herd_immunity}%')

    # noinspection PyUnboundLocalVariable
    results.append(f'{herd_immunity}%')
    return results


def three_month():
    results = list()
    # print("Three-month gap:")
    no_doses, one_month, two_months, three_months, two_doses = TOTAL_POP, 0, 0, 0, 0
    total_supply = 0
    for i in range(1000):
        new_supply = MONTHLY_SUPPLY if total_supply < (TOTAL_POP * 2) else 0

        if not new_supply:
            break

        total_supply += new_supply

        # Give second dose to as many as you can who are at three-month mark
        if three_months > 0:
            shots_given = min(three_months, new_supply)
            two_doses += shots_given
            three_months -= shots_given
            new_supply -= shots_given

        # Progress the once-vaccinated groups by one month
        three_months += two_months
        two_months = one_month
        one_month = 0

        # Use what you have left to give first doses
        if no_doses > 0:
            shots_given = min(no_doses, new_supply)
            no_doses -= shots_given
            one_month = shots_given
            new_supply -= shots_given

        # If there is still supply left, give it to the people waiting longest
        if new_supply > 0:
            shots_given = min(three_months, new_supply)
            three_months -= shots_given
            two_doses += shots_given
            new_supply -= shots_given

        # If there is still supply left, give it to the people waiting longest
        if new_supply > 0:
            shots_given = min(two_months, new_supply)
            two_months -= shots_given
            two_doses += shots_given
            new_supply -= shots_given

        # If there is still supply left, give it to the people waiting longest
        if new_supply > 0:
            shots_given = min(one_month, new_supply)
            one_month -= shots_given
            two_doses += shots_given
            new_supply -= shots_given

        herd_immunity = (((one_month + two_months + three_months) * ONE_DOSE_EFFICACY) +
                         (two_doses * TWO_DOSE_EFFICACY)) / TOTAL_POP
        herd_immunity = int((herd_immunity * 100) + 0.5)
        results.append(f'{herd_immunity}%')

    # noinspection PyUnboundLocalVariable
    results.append(f'{herd_immunity}%')
    return results


# noinspection PyUnusedLocal
def __main():
    # print(worried())
    # print(confident())
    # print(three_month())
    cols = dict(Worried=worried(), Confident=confident(), Three_Month=three_month())
    df = pandas.DataFrame.from_dict(cols)
    df.index += 1
    print(df)


if __name__ == '__main__':
    __main()


