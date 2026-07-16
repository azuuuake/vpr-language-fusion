# Nordland prompted-caption comparison for low-LU failures

This diagnostic compares original captions and prompted captions for low-LU Nordland cases where visual top-1 is correct but language-only top-1 is wrong.

## Aggregate summary

- **n_failures**: 13

- **mean_orig_query_tokens**: 4.923076923076923

- **mean_prompt_query_tokens**: 7.461538461538462

- **mean_query_token_gain**: 2.5384615384615383

- **mean_orig_positive_tokens**: 5.0

- **mean_prompt_positive_tokens**: 8.307692307692308

- **mean_positive_token_gain**: 3.3076923076923075

- **mean_orig_query_wrong_jaccard**: 0.6091880341880341

- **mean_prompt_query_wrong_jaccard**: 0.3171983998907076

- **mean_orig_query_positive_jaccard**: 0.12307692307692308

- **mean_prompt_query_positive_jaccard**: 0.23504193142202187


## Example cases


### Query 142

- LU: 0.804366

- Positive DB indices: [115, 116, 117, 118, 119]


**Original captions**

- Query: a train traveling down the tracks in the snow

- True positive: a view of a train track with cars parked on it

- Wrong language top-1: a train traveling down the tracks in the mountains


**Prompted captions**

- Query: A railway track with a curve in the terrain and a building in the background.

- True positive: A railway track with a curve in the terrain.

- Wrong language top-1: A railway track with a mountain in the background.


### Query 143

- LU: 0.801166

- Positive DB indices: [116, 117, 118, 119, 120]


**Original captions**

- Query: a train traveling down the tracks in the snow

- True positive: a view of a train track with cars parked on it

- Wrong language top-1: a train traveling down the tracks in the mountains


**Prompted captions**

- Query: A mountain road with a red building on the right side.

- True positive: A railway track with a curve in the terrain and a bridge in the background.

- Wrong language top-1: A railway track with a mountain in the background.


### Query 151

- LU: 0.756797

- Positive DB indices: [124, 125, 126, 127, 128]


**Original captions**

- Query: a car driving down a snowy road in the mountains

- True positive: a view of a train track with cars parked on it

- Wrong language top-1: a train traveling down the tracks in the mountains


**Prompted captions**

- Query: A mountain road with a hill on the left and a bridge on the right.

- True positive: A railway track with a curve in the terrain.

- Wrong language top-1: A railway track with a mountain in the background.


### Query 175

- LU: 0.800361

- Positive DB indices: [148, 149, 150, 151, 152]


**Original captions**

- Query: a view of a snowy road in the mountains

- True positive: a train traveling down the tracks in a rural area

- Wrong language top-1: a view of a train track with mountains in the background


**Prompted captions**

- Query: A railway track runs through the track corridor that is curved and surrounded by hills and flat land.

- True positive: A railway track with a curve in the terrain, hills on the left and right side, and a bridge in the background.

- Wrong language top-1: A railway track running through a mountainous area with a forest on the right side.


### Query 185

- LU: 0.787162

- Positive DB indices: [158, 159, 160, 161, 162]


**Original captions**

- Query: an image of a black screen with a white text on it

- True positive: a train traveling down the tracks next to a body of water

- Wrong language top-1: a black background with a white text on it


**Prompted captions**

- Query: A black and white photo of A railway track with a hill on the left side and a bridge on the right side.

- True positive: A railway track runs through the track corridor that is curved and surrounded by hills, flat land, and slopes.

- Wrong language top-1: A black and white photo of A railway track with a curve in the middle.


### Query 189

- LU: 0.773081

- Positive DB indices: [161, 162, 163, 164, 165]


**Original captions**

- Query: an image of a black background with white text

- True positive: a train traveling down the tracks next to a body of water

- Wrong language top-1: a black background with a white text on it


**Prompted captions**

- Query: A black and white photo of A railway track with a hill on the left side and a bridge on the right side.

- True positive: A railway track runs through the track corridor that is curved and surrounded by hills and flat land.

- Wrong language top-1: A black and white photo of A railway track with a curve in the middle.


### Query 193

- LU: 0.806102

- Positive DB indices: [165, 167, 168, 169, 170]


**Original captions**

- Query: a train traveling along a snowy mountain side

- True positive: a view from a train window looking down the tracks

- Wrong language top-1: a train traveling down the tracks in the mountains


**Prompted captions**

- Query: A railway track with a curve in the terrain and a bridge in the background.

- True positive: A railway track running through a mountainous area with trees on the sides.

- Wrong language top-1: A railway track with a curve in the middle, surrounded by hills and a tunnel.


### Query 202

- LU: 0.707369

- Positive DB indices: [174, 175, 176, 178, 179]


**Original captions**

- Query: a view of a train going over a bridge

- True positive: a train going down the tracks in the mountains

- Wrong language top-1: a view of a train going over a bridge


**Prompted captions**

- Query: A railway track with a bridge and a mountain in the background.

- True positive: A railway track with a bridge and a tunnel in the background.

- Wrong language top-1: A railway track runs through the track corridor that is curved and surrounded by hills and vegetation.


### Query 338

- LU: 0.779535

- Positive DB indices: [110, 122, 133, 144, 155]


**Original captions**

- Query: a train traveling through the snow covered mountains

- True positive: a train traveling down the tracks in the woods

- Wrong language top-1: a train traveling down the tracks in the mountains


**Prompted captions**

- Query: A railway track with a curve in the terrain and a bridge in the background.

- True positive: A railway track runs through the track corridor that is curving around a hillside, with a forest nearby.

- Wrong language top-1: A railway track with a mountain in the background.


### Query 348

- LU: 0.780441

- Positive DB indices: [210, 221, 233, 244, 255]


**Original captions**

- Query: a train is traveling through the snow covered forest

- True positive: a train traveling down the tracks next to a river

- Wrong language top-1: a train traveling through a forest filled with trees


**Prompted captions**

- Query: A railway track with a curve in the terrain.

- True positive: A railway track runs through the track corridor that is surrounded by hills, a lake, and trees.

- Wrong language top-1: A railway track with a curve in the terrain, surrounded by trees and hills.


### Query 359

- LU: 0.793349

- Positive DB indices: [321, 332, 334, 335, 336]


**Original captions**

- Query: a train traveling through the snow covered forest

- True positive: a train going through a tunnel in a tunnel

- Wrong language top-1: a train traveling through a forest filled with trees


**Prompted captions**

- Query: A railway track runs through the track corridor that is curved and surrounded by hills and flat land.

- True positive: A railway travels through a tunnel that is surrounded by hills and vegetation.

- Wrong language top-1: A railway track running through a forest with a hill on the left and right side.


### Query 378

- LU: 0.781223

- Positive DB indices: [1, 2, 3, 4, 5]


**Original captions**

- Query: a train going through a tunnel in the dark

- True positive: a train traveling down the tracks in the woods

- Wrong language top-1: a train going through a tunnel in the dark


**Prompted captions**

- Query: A railway travels through a tunnel that is surrounded by hills and flat land.

- True positive: A railway track running through a forest with hills and flat land on both sides.

- Wrong language top-1: A railway track with a tunnel on the left side and a bridge on the right side.
