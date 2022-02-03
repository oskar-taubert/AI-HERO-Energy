import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import date
from tqdm import tqdm
import torch
pd.options.mode.chained_assignment = None

verbose = False


df = pd.read_csv("/hkfs/work/workspace/scratch/bh6321-energy_challenge/data/train.csv", delimiter=",")

df["Year"] = [datetime.strptime(i, "%Y-%m-%d %H:%M:%S").year for i in df["Time [s]"]]
df["Date"] = [i.split(" ")[0] for i in df["Time [s]"]]
df["Week"] = [date(int(i.split("-")[0]), int(i.split("-")[1]), int(i.split("-")[2])).isocalendar()[1] for i in df["Date"]]



continous_weeks = []
len_first_week = df[(df.Year == 2015) & (df.Week == 1) & (df.City == "bs")].shape[0]
len_normal_week = df[(df.Year == 2015) & (df.Week == 2) & (df.City == "bs")].shape[0]
week_dif = len_normal_week - len_first_week

count = 1
for i in range(df[df.City == "bs"].shape[0]):
    if (i + week_dif) % len_normal_week == 0:
        count += 1
    continous_weeks.append(count)


df["Continous_week"] = 14*continous_weeks

# rolling 14 day window with city specific normalization
# averaging over all 14 city values

weeklevel = []
calenderweeklist = []
var_weeklevel = []

citynames = []
all_teh_data = []

for week in tqdm(df.Continous_week.unique()[:-1]):
    df_week = df.loc[(df.Continous_week == week) | (df.Continous_week == week+1)]
    calenderweek = df_week.Week.unique()[0]
    calenderweeklist.append(calenderweek)
    citylevel = []
    for city in df_week.City.unique():
        citynames.append(city)
        df_city = df_week[df_week.City == city]
        mean_, sd_ = df_city["Load [MWh]"].median(), df_city["Load [MWh]"].std()
        upper, lower = mean_ + 3.5*sd_, mean_ - 3.5*sd_

        sum_ = ((df_city["Load [MWh]"] < lower) | (df_city["Load [MWh]"] > upper)).sum()
        df_city.loc[(df_city["Load [MWh]"] < lower), "Load [MWh]"] = lower
        df_city.loc[(df_city["Load [MWh]"] > upper), "Load [MWh]"] = upper
        
        if verbose and sum_ > 0:
            print(f"Clipped {sum_} values for city {city} and week {week}.")
        if verbose and sum_ > 5:
            print(f"Warning for week {week} and city {city}. There were {sum_} outliers.")
        min_, max_ = df_city[df_city.Continous_week == week]["Load [MWh]"].min(), df_city[df_city.Continous_week == week]["Load [MWh]"].max()

        df_city["Load_norm"] = [(df_city["Load [MWh]"].iloc[i]-min_)/(max_ - min_)*2 - 1 for i in range(df_city.shape[0])]

        week1 = df_city[df_city.Continous_week == week]["Load_norm"].mean()
        week2 = df_city[df_city.Continous_week == week+1]["Load_norm"].mean()
        delta_ = week2 - week1
        citylevel.append(delta_)
    all_teh_data.append([citylevel])
    weeklevel.append(np.mean(citylevel))
    var_weeklevel.append(np.std(citylevel))



df_deltas = pd.DataFrame({"Calenderweek":calenderweeklist, "Delta":weeklevel})

df_result = df_deltas.groupby(["Calenderweek"])["Delta"].agg([np.mean, np.std]).reset_index()
df_result["delta_cumsum"] = df_result["mean"].cumsum()
print(df_result)

seasonal_delta = torch.tensor(df_result["mean"], dtype=torch.float)
print(seasonal_delta)

fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
ax1.plot(df_result["mean"])
ax1.set_ylabel("Pre-week Delta")
ax1.grid()
ax2.plot(df_result["delta_cumsum"])
ax2.set_ylabel("Cumulative Sum")
ax2.set_xlabel("Calenderweek")
ax2.grid()
plt.subplots_adjust(wspace=0, hspace=0)

plt.savefig("deltas.png")
plt.close()


years = np.array([i-2015 for i in df.Year.unique()])
slopes = []
sections = []
for city in df.City.unique():
    meansies = []
    for year in df.Year.unique():
        meansy = df[(df.City == city) | (df.Year == year)]["Load [MWh]"].mean()
        meansies.append(meansy)
    coefs = np.polyfit(years, meansies, deg=1)
    print(coefs)
    
    slopes.append(coefs[0])
    sections.append(coefs[1])
cosmic_slope = torch.tensor(np.mean(slopes), dtype=torch.float)
cosmic_section = torch.tensor(np.mean(sections), dtype=torch.float)

print(cosmic_slope)
print(cosmic_section)

dict_parameters = {"seasonal_delta":seasonal_delta, "cosmic_slope":cosmic_slope, "cosmic_intersection":cosmic_section}
torch.save(dict_parameters, "naive_parameters.pt")


