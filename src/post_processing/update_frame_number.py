import csv

input_file = '../../data/video_2/bbox_video2p2.csv'
output_file = '../../data/video_2/bbox_video2p2comp.csv'

addition_value = 75306

with open(input_file, newline='') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for row in reader:
        try:
            row['Event frame'] = str(int(row['Event frame']) + addition_value)
        except ValueError:
            pass
        writer.writerow(row)

print(f"Modified CSV saved to {output_file}")
