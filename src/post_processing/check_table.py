import csv

def find_empty_table_boxes(csv_path):
    empty_frames = []
    
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['Table boxes'].strip() == '[]':
                empty_frames.append(row['Event frame'])
    
    return empty_frames

video = 2 
path_to_csv = f"../../data/video_{video}/merged_output_video{video}.csv"
frames = find_empty_table_boxes(path_to_csv)

if frames:
    print("Event frames with empty 'Table boxes':")
    for frame in frames:
        print(frame)
else:
    print("No rows found with empty 'Table boxes'.")
