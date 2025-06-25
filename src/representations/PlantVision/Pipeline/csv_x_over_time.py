"""
The purpose of this script is to create a new csv file that contains a timestamp column, and 
columns for a certain trait of each of the plants.

example:
csv columns before:
timestamp, sample, genotype, estimated_object_count, in_bounds, area, ..., plant_id

command: python3 csv_x_over_time.py csv_file_with_timestamp.csv area

csv columns after:
timestamp, plant 0 (WT), plant 1 (phyA), ..., plant 47 (phot1/2)

update: feb 22 2024 I now start counting plants at 1
"""
import csv
import sys


def calculate_num_plants(csv_file):
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        plant_id_index = header.index('plant_id')
        num_plants = 0
        for row in reader:
            try:
                plant_id = int(row[plant_id_index])
            except Exception:
                continue
            if plant_id > num_plants:
                num_plants = plant_id
    return num_plants

def csv_x_over_time(csv_file, trait):
    NUM_PLANTS = calculate_num_plants(csv_file)

    outfile = csv_file.replace('.csv', f'_{trait}_over_time.csv')
    outfile_contents = {}
    genotype_map = ["N/A"] * NUM_PLANTS
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        trait_index = header.index(trait)
        plant_id_index = header.index('plant_id')
        genotype_index = header.index('genotype')
        timestamp_index = header.index('timestamp')
        for row in reader:
            try:
                timestamp = row[timestamp_index]
                if timestamp not in outfile_contents:
                    outfile_contents[timestamp] = [""] * NUM_PLANTS
                plant_id = int(row[plant_id_index])-1
                genotype = row[genotype_index]
            except Exception:
                continue

            genotype_map[plant_id] = genotype
            outfile_contents[timestamp][plant_id] = row[trait_index]

        with open(outfile, 'w', newline='') as out:
            writer = csv.writer(out)
            header = ["timestamp"]
            for i in range(0, NUM_PLANTS):
                header.append(f"plant {i+1} ({genotype_map[i]})")
            writer.writerow(header)
            for timestamp in sorted(outfile_contents.keys()):
                row = [timestamp] + outfile_contents[timestamp]
                writer.writerow(row)

    return outfile


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        for arg in sys.argv:
            for other_arg in sys.argv:
                if arg != other_arg and not (arg.endswith('.py') or other_arg.endswith('.py')):
                    if arg.endswith('.csv') and not other_arg.endswith('.csv'):
                        csv_file = arg
                        trait = other_arg
                        csv_x_over_time(csv_file, trait)
                    elif not arg.endswith('.csv') and other_arg.endswith('.csv'):
                        csv_file = other_arg
                        trait = arg
                        csv_x_over_time(csv_file, trait)
    else:
        print("\nUsage: python create_x_over_time_csv.py <csv_file> <trait>\n")
