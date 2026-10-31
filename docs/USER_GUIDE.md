# User guide

This guide explains every part of the HazeCast SG screen. You don't need to know anything about machine learning to use it.

![The whole app on a laptop](img/app_desktop.png)

## What PM2.5 means

PM2.5 is the amount of very fine dust and smoke in the air. The particles are smaller than 2.5 micrometres, about thirty times thinner than a human hair, so they get deep into your lungs. It is measured in micrograms per cubic metre of air, written µg/m³. Lower is better. On a normal day in Singapore it sits around 10 to 30. During bad haze it can go above 200.

## 1. Pick a time

The four tabs at the top choose what the whole screen shows.

**Now** shows the latest real readings from NEA's stations. **In 6 hrs**, **In 12 hrs** and **Tomorrow** show our forecasts for that many hours ahead. Tomorrow means 24 hours from the latest reading, so if the last reading was at 11 pm, Tomorrow is 11 pm the next day.

## 2. Say who's going out

**Everyone** uses NEA's bands: green up to 55, yellow from 56 to 150, red from 151.

**Children, Elderly and Asthmatic Patients** is stricter, because these groups feel the haze sooner. Green only goes up to 35, yellow means keep outdoor time short and wear a mask, and red starts at 56. The map colours and the advice change when you switch. This stricter rule is our own cautious choice, not an official NEA one.

The app remembers your choice next time you open it.

## 3. Read the map

Each of the five NEA regions is coloured by what you should do there. The white badge shows the PM2.5 number. The box in the corner of the map is the key.

Your region has a dark outline and a small **You** tag. To change it, tap a region, or tap **Use my location** (shown as **My location** on smaller phones) and allow location access when your browser asks. The app then finds the region closest to you. If you are outside Singapore, it asks you to tap the region you are heading to instead.

Location works best when the app is opened through a web address, like localhost or GitHub Pages. Your location never leaves your device. The app only uses it to pick the nearest region.

## 4. The summary

Under the map, the otter mascot and a short headline sum up the whole island:

| Headline | What it means |
|---|---|
| Outdoors! Enjoy the sun | Every region is green |
| Outdoors, with a mask in places | Some regions are yellow, none red |
| Mask on if you head out | Every region is yellow |
| Depends where you are | At least one region is red |
| Stay indoors | Every region is red |

The line under the headline gives the number and advice for your own region.

## 5. Best time to go out

This card puts the four time slots side by side for your region. The slot with a dark border is the best one: the first slot that is green, or if none are green, the one with the lowest number. The times are rough, so "around 11 am" means the forecast for that hour, not a guaranteed window.

## 6. How sure are we?

No forecast is perfect, so this card is honest about how ours did.

Before we trusted the model, we hid two and a half years of past data from it (April 2024 to October 2026) and asked it to forecast that period. The ring shows how many of the truly unhealthy hours it warned about. For the 6-hour forecast that was about 8 in 10. Forecasts further ahead are harder, so Tomorrow catches about 5 in 10.

The second number, "right when it warned", tells you how often a warning turned out to be a real unhealthy hour. We deliberately set the model to lean towards warning, because telling you to carry a mask on a fine day is better than letting you walk into haze.

On the **Now** tab the ring says **Real**, since those are measured readings, not a forecast.

The line at the bottom is the model's health check. Every night the pipeline checks whether the incoming data still looks normal and whether recent forecasts were accurate. Green means all good. Yellow means something looks off and a person should look. Orange means the model is due to be retrained. Red means the input data has a problem.

## 7. Live and Demo

The label under the title says **Live** when the app is reading the latest numbers from our server, and **Demo** when it is showing a saved snapshot.

The app switches to Demo by itself if the server can't be reached, for example on GitHub Pages, which has no server. A yellow note explains this. You can also switch by hand with the Live and Demo buttons at the bottom. The demo is real data from 11 pm on 16 October 2026, a night with elevated haze.

If the live forecast was made from data more than three hours old, a yellow note warns you. This happens when a data source is late.

## Common questions

**Why do Now and the forecasts sometimes disagree a lot?**
Haze can change fast when the wind shifts. The forecast also leans high on purpose, as explained in "How sure are we?".

**Why is my region yellow when it looks clear outside?**
PM2.5 particles are too small to see at lower levels. You usually only see haze once it is well into the yellow range.

**Where do the numbers come from?**
PM2.5 from NEA through data.gov.sg, wind and rain from Open-Meteo, and fire hotspots from NASA FIRMS.

**Is this official advice?**
No. It is a student project. For official advice, check [haze.gov.sg](https://www.haze.gov.sg).

## For developers

The app is one file, `serving/static/index.html`, with an identical copy at `docs/index.html` for GitHub Pages. It reads three API endpoints:

| Endpoint | What it gives the app |
|---|---|
| `/forecast/hourly` | The 6, 12 and 24 hour forecasts for each region |
| `/context` | Latest readings, test accuracy and the health check |
| `/config` | The NEA threshold the model was tested against |

`/context` is built by `src/hazecast/context.py` from files the pipeline already writes, so it adds no extra work to the hourly run. If `/context` fails, the forecast still shows. If `/forecast/hourly` fails, the app falls back to the demo.

When you change the page, copy it over `docs/index.html` too. `tests/test_api.py` checks that the two match and that the page only calls endpoints that exist.
