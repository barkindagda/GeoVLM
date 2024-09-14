import pandas as pd

# Load the CSV file
df = pd.read_csv('Path to the csv file')

# List of questions to be used in the description
questions = [
    "What is the predominant environment in the image? (urban/suburban/rural/highway/industrial/natural/dense forestation/water body/mixed)",
    "What type of road layout is visible in the image? (grid pattern/winding roads/roundabout/dead-end streets/highway/none/mixed patterns)",
    "What specific environmental features are visible in the image? (Residential buildings / Commercial areas / Factories / Farms / Green spaces / Parks / Rivers / Lakes / Forests / Beaches / Cliffs / Hills / Open fields / Vacant lots / None / Other)",
    "What kind of distinct road features are present in the image? (none/simple intersections/complex junctions/overpasses/roundabouts/traffic circles)",
    "What types of buildings are most common in the image? (residential houses/apartment buildings/commercial buildings/industrial facilities/public buildings/mixed/no buildings/other)",
    "What is the condition of the vegetation in the image? (None/dense forests/parklands/sparse vegetation/agricultural fields/barren land/ornamental gardens)",
    "What distinctive features are present in the image? (None / Natural Landmarks / Historical Buildings / Modern Structures / Sporting Facilities / Water Bodies / Parks / Urban Art / Monuments / Industrial Facilities / Other)",
    "What is the architecture style of the buildings in the image ? (None/traditional/modern/industrial/mixed/historical)",
    "What type of transportation features can be seen in the image? (None/train tracks/airports/ports/tram lines/bus stations)",
    "What kind of large, open spaces are there in the picture? (None/fields/empty lots/forests/car parks/urban squares/golf course/public garden/playgrounds/sports field)",
    "What is the overall layout of the area observed in the image? (organized/disorganized/mixed/regular pattern/irregular pattern/none/chaotic)",
    "What are the unique patterns in roads or buildings in the image? (none/linear patterns/radial patterns/grid patterns/irregular patterns/circular patterns)",
    "What is the predominant color of the roofs in the image? (red/brown/grey/white/green/other/none/multi-colored)",
    "What is the predominant color of the roads in the image? (black/grey/red/yellow/other/none/multi-colored)",
    "What other notable color features are present in the image? (green areas/water bodies/colored buildings/sports fields/none/colorful gardens)",
    "What type of main road is visible in the image? (none/single-lane road/multi-lane road/highway/expressway)",
    "What road markings are present in the image? (None / Zebra crossings / Chevrons / White lines / Yellow lines / Double yellow lines / Arrows / Stop lines / Crosswalks / Bicycle lanes / Bus lanes / Hatched markings / Box junctions / School crossings / Speed limit markings / Other)",
    "What are the predominant colors of the road markings in the image? (White / Yellow / Red / Blue / Green / None / Other / Multi-colored)",
    "Are there any of the following road structures visible in the image? (None / Bridge / Underpass / Overpass / Tunnel / Flyover / Pedestrian crossing bridge / Roundabout / Highway interchange / Railway crossing / Other)",
    "How would you describe the orientation of the roads in the image? (Straight highway / Single road / Multiple parallel roads / Multiple roads converging / Multiple roads diverging / Intersection / Roundabout / Serpentine or winding road / T-junction / Crossroads / Forked road / Overpass/underpass systems / Cul-de-sac / One-way street / Pedestrian-only path / Bicycle lane / None / Other)",
    "What are the predominant types of surrounding vehicles in the image? (Cars / Trucks / Bicycles / Motorcycles / Public Transport / None)",
    "What is the directional layout of the road junction in the image? (none/left turn only / right turn only / both left and right turns / four-way intersection / roundabout / multiple direction options / complex multi-way junction / other)",
    "What is the width of the road? (None/narrow/medium/wide/multiple lanes/variable widths)",
    "Are there any traffic lights present along the road? (yes/no)",
    "Are there any billboard signs on the road indicating directions or destinations? (Yes / No)",
    "Is there a rest area or service station visible in the image? (yes/no)",
    "What type of service facility is visible in the image? (None/Petrol station / Supermarket / Restaurant / Hotel)",
    "Are any sports courts visible? (None/basketball/tennis/football)",
    "Does the road have a hard shoulder or emergency lane? (yes/no)",
    "Is there a pedestrian area like a sidewalk or footpath alongside the road? (yes/no)"
]

# Function to generate description for each row using all questions
def generate_description(row):
    description = (
        f"The image shows a {row[questions[0]]} area with a {row[questions[1]]} road layout, "
        f"featuring {row[questions[2]]} such as {row[questions[3]]}. "
        f"The buildings predominantly visible are {row[questions[4]]}, "
        f"with vegetation described as {row[questions[5]]}. "
        f"Distinctive features include {row[questions[6]]}, "
        f"and the architecture style is {row[questions[7]]}. "
        f"Transportation features visible include {row[questions[8]]}, "
        f"with open spaces like {row[questions[9]]}. "
        f"The area is {row[questions[10]]} in layout, with {row[questions[11]]} patterns observed in the roads and buildings. "
        f"The roofs are predominantly {row[questions[12]]} in color, while the roads are {row[questions[13]]} with {row[questions[14]]}. "
        f"The main road visible is a {row[questions[15]]}, with road markings such as {row[questions[16]]} in {row[questions[17]]}. "
        f"Road structures include {row[questions[18]]}, and the orientation of the roads is described as {row[questions[19]]}. "
        f"Surrounding vehicles are mainly {row[questions[20]]}. "
        f"The road junction allows for {row[questions[21]]} traffic flow, with a road width of {row[questions[22]]}. "
        f"Traffic lights present: {row[questions[23]]}, with billboard signs: {row[questions[24]]}. "
        f"A rest area or service station is {row[questions[25]]} visible, offering {row[questions[26]]}. "
        f"Sports courts such as {row[questions[27]]} are visible, and the road includes a hard shoulder: {row[questions[28]]}. "
        f"Finally, there is {row[questions[29]]} a pedestrian area like a sidewalk or footpath."
    )
    return description

# Apply the function to each row in the dataframe
df['caption'] = df.apply(generate_description, axis=1)

# Select only the relevant columns
output_df = df[['image_name', 'caption']]

# Save the results to a new CSV file
output_df.to_csv('./captions_(query)_or_(satellite).csv', index=False)
