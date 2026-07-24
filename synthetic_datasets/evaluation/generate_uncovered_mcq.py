import argparse
import ast
import random
import pandas as pd

def add_article(obj_name):
    """Add appropriate indefinite article (a/an) to singular object nouns."""
    obj_name = obj_name.strip()
    if not obj_name:
        return obj_name
    words = obj_name.split()
    first_word = words[0].lower()
    if first_word in ['a', 'an', 'the', 'some', 'several']:
        return obj_name
    if first_word[0] in ['a', 'e', 'i', 'o', 'u']:
        return f"an {obj_name}"
    return f"a {obj_name}"

def parse_args():
    parser = argparse.ArgumentParser(description="Generate template-based MCQs for uncovered negative objects in top sorted images.")
    parser.add_argument('--retrieval_file', type=str, default='COCO_val_retrieval.csv', help='Path to retrieval CSV file')
    parser.add_argument('--mcq_file', type=str, default='COCO_val_mcq_llama3.1_rephrased.csv', help='Path to existing MCQ CSV file')
    parser.add_argument('--output_file', type=str, default='COCO_val_mcq_top100_uncovered.csv', help='Path to save generated MCQ dataset')
    parser.add_argument('--top_n', type=int, default=100, help='Number of sorted images to process')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)

    print(f"Loading retrieval dataset from: {args.retrieval_file}")
    ret_df = pd.read_csv(args.retrieval_file)

    print(f"Loading existing MCQ dataset from: {args.mcq_file}")
    mcq_df = pd.read_csv(args.mcq_file, encoding='latin1')

    # Sort retrieval dataset by filepath
    ret_df_sorted = ret_df.sort_values(by='filepath').reset_index(drop=True)
    top_images_df = ret_df_sorted.head(args.top_n)

    print(f"Processing top {len(top_images_df)} images sorted by filepath...")

    templates = ["positive", "negative", "hybrid"]
    mcq_data = []

    total_images_processed = 0
    total_uncovered_objects = 0

    for idx, row in top_images_df.iterrows():
        filepath = row['filepath']
        pos_objs = ast.literal_eval(row['positive_objects']) if isinstance(row['positive_objects'], str) else row['positive_objects']
        neg_objs = ast.literal_eval(row['negative_objects']) if isinstance(row['negative_objects'], str) else row['negative_objects']

        if not isinstance(pos_objs, list):
            pos_objs = []
        if not isinstance(neg_objs, list):
            neg_objs = []

        total_images_processed += 1

        # Check existing MCQ rows for this filepath to find covered negative objects
        img_mcqs = mcq_df[mcq_df['image_path'] == filepath]
        used_negs = set()
        if not img_mcqs.empty:
            for _, mcq_row in img_mcqs.iterrows():
                text = ' '.join([str(mcq_row[c]) for c in ['caption_0', 'caption_1', 'caption_2', 'caption_3'] if pd.notna(mcq_row[c])]).lower()
                for obj in neg_objs:
                    if obj.lower() in text:
                        used_negs.add(obj)

        uncovered_negs = [obj for obj in neg_objs if obj not in used_negs]
        total_uncovered_objects += len(uncovered_negs)

        if not uncovered_negs:
            continue

        for N in uncovered_negs:
            art_N = add_article(N)

            # Pick positive objects A and B
            if len(pos_objs) >= 2:
                A, B = random.sample(pos_objs, 2)
            elif len(pos_objs) == 1:
                A = pos_objs[0]
                B = pos_objs[0]
            else:
                A = "object"
                B = "object"

            art_A = add_article(A)
            art_B = add_article(B)

            for tmpl in templates:
                if tmpl == "positive":
                    if len(pos_objs) >= 2:
                        right_answer = f"This image features {art_A} and {art_B}."
                    else:
                        right_answer = f"This image features {art_A}."
                    wrong_answer_1 = f"This image features {art_N}, with no {B} in sight."
                    wrong_answer_2 = f"This image features {art_N}."
                    wrong_answer_3 = f"A {A} is not present in this image."

                elif tmpl == "negative":
                    # Randomize negative phrasing to match COCO_val_mcq_llama3.1_rephrased.csv diversity
                    neg_phrasings = [
                        f"There is no {N} in this image.",
                        f"A {N} is not included in this image.",
                        f"This image does not feature {art_N}.",
                        f"No {N} is present in this image."
                    ]
                    right_answer = random.choice(neg_phrasings)
                    wrong_answer_1 = f"This image features {art_N}, with no {B} in sight."
                    wrong_answer_2 = f"This image shows {art_N}."
                    wrong_answer_3 = f"No {B} is present in this image."

                elif tmpl == "hybrid":
                    right_answer = f"This image features {art_B}, with no {N} in sight."
                    wrong_answer_1 = f"This image features {art_N}, but no {B} is visible."
                    wrong_answer_2 = f"This image features {art_N}."
                    wrong_answer_3 = f"There is no {B} in this image."

                answers = [right_answer, wrong_answer_1, wrong_answer_2, wrong_answer_3]

                mcq_data.append({
                    'image_path': filepath,
                    'correct_answer': 0,
                    'caption_0': answers[0],
                    'caption_1': answers[1],
                    'caption_2': answers[2],
                    'caption_3': answers[3],
                    'correct_answer_template': tmpl
                })

    output_df = pd.DataFrame(mcq_data)
    cols = ['image_path', 'correct_answer', 'caption_0', 'caption_1', 'caption_2', 'caption_3', 'correct_answer_template']
    output_df = output_df[cols]

    output_df.to_csv(args.output_file, index=False)

    print("\n--- Execution Summary ---")
    print(f"Total Target Images Processed: {total_images_processed}")
    print(f"Total Uncovered Negative Objects Found: {total_uncovered_objects}")
    print(f"Total MCQ Questions Generated: {len(output_df)}")
    print(f"Output saved to: {args.output_file}")

if __name__ == '__main__':
    main()
