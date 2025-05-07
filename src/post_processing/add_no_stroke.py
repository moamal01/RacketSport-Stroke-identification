import json
import random

video = 3

with open(f"data/events/events_markup{video}.json", "r") as f:
    data = json.load(f)

# Step 1: Add ±5 around every 10th valid (non-skip) event
skip_values = {"net", "bounce", "empty_event"}
sorted_items = sorted(data.items(), key=lambda x: int(x[0]))
selected_items = sorted_items[::10]

# Keep track of new keys to avoid duplicates
new_entries = {}
existing_keys = set(data.keys())

for key_str, value in selected_items:
    if value not in skip_values:
        key = int(key_str)
        for offset in [-5, 5]:
            new_key = str(key + offset)
            if new_key not in existing_keys:
                new_entries[new_key] = "no_stroke"
                existing_keys.add(new_key)

# Step 2: Add 25 random unique "no_stroke" keys
all_numeric_keys = set(int(k) for k in data.keys()) | set(int(k) for k in new_entries.keys())
min_key = min(all_numeric_keys)
max_key = max(all_numeric_keys)

added_random = 0
while added_random < 25:
    rand_key = random.randint(min_key, max_key)
    rand_key_str = str(rand_key)
    if rand_key_str not in existing_keys:
        new_entries[rand_key_str] = "no_stroke"
        existing_keys.add(rand_key_str)
        added_random += 1

# Combine all new entries into the original data
data.update(new_entries)
# Sort the updated data by key numerically
sorted_data = dict(sorted(data.items(), key=lambda x: int(x[0])))

# Save the sorted data
with open(f"data/extended_events/events_markup{video}.json", "w") as f:
    json.dump(sorted_data, f, indent=4)

print(f"Added {len(new_entries)} 'no_stroke' entries (including 25 random ones).")
