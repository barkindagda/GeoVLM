import os
import csv
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import openai
import pandas as pd

image_folder = 'path/to/image/folder'
final_output_csv = 'path/to/final/output/csv/with/descriptions.csv'

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

def VQA_to_description(image_path, questions, model, vis_processors, txt_processors, writer):
    """
    Process a single image to generate Q&As and then use those to generate a description.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_image = Image.open(image_path).convert('RGB')
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    answers = []
    for question in questions:
        processed_question = txt_processors["eval"](question)
        generative_answer = model.predict_answers(
            samples={"image": image, "text_input": processed_question},
            inference_method="generate"
        )
        answers.append(generative_answer[0])
    
    # Construct the description prompt based on generated Q&As
    prompt = construct_prompt(questions, answers)
    description = MoE_agent(prompt)
    
    # Write the filename and description to the final CSV
    writer.writerow([os.path.basename(image_path), description])
  
def construct_prompt(questions, answers):
    """
    Constructs a prompt for description generation based on questions and answers.
    """
    prompt = "<s>[INST] Generate a description for the image based on the following information:\n"
    for question, answer in zip(questions, answers):
        prompt += f"{question}: {answer}\n"
    prompt += "[/INST]"
    return prompt
  
def MoE_agent(prompt):
    response = client.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        prompt=prompt,
        max_tokens=300
    )
    return response.choices[0].text.strip()
  
def main(image_folder,final_output_csv,questions):
      """
    Main function to process images and generate descriptions without an intermediate CSV.
    """
    # Initialize model and processors here (similar to the first script)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip_vqa", model_type="vqav2", is_eval=True, device=device
    )

    with open(final_output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Description"])

        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_folder, filename)
                VQA_to_description(image_path, questions, model, vis_processors, txt_processors, writer)

if __name__ == "__main__":
    main(image_folder, final_output_csv, questions)

