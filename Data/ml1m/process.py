import pickle
import numpy as np
from scipy.sparse import coo_matrix
from collections import defaultdict

# ============================================================
# Step 1: Read ratings.dat and filter positive interactions
# Format: UserID::MovieID::Rating::Timestamp
# Keep ratings >= 4 as positive (implicit feedback)
# ============================================================
ratings_path = "./ml-1m/ratings.dat"
print("Reading ratings.dat ...")

user_items = defaultdict(list)  # user -> [(item, timestamp), ...]
all_users = set()
all_items = set()

with open(ratings_path, "r") as f:
    for line in f:
        parts = line.strip().split("::")
        user_id = int(parts[0])
        item_id = int(parts[1])
        rating = float(parts[2])
        timestamp = int(parts[3])
        if rating >= 4.0:
            user_items[user_id].append((item_id, timestamp))
            all_users.add(user_id)
            all_items.add(item_id)

print(f"Total positive interactions (rating>=4): {sum(len(v) for v in user_items.values())}")
print(f"Users with positive interactions: {len(all_users)}")
print(f"Items with positive interactions: {len(all_items)}")

# ============================================================
# Step 2: Filter users/items with too few interactions (10-core)
# ============================================================
print("\nApplying 10-core filtering ...")
min_user_inter = 10
min_item_inter = 10

for iteration in range(100):
    item_count = defaultdict(int)
    for uid, items in user_items.items():
        for iid, ts in items:
            item_count[iid] += 1

    valid_items = {iid for iid, cnt in item_count.items() if cnt >= min_item_inter}

    new_user_items = defaultdict(list)
    for uid, items in user_items.items():
        filtered = [(iid, ts) for iid, ts in items if iid in valid_items]
        if len(filtered) >= min_user_inter:
            new_user_items[uid] = filtered

    prev_len = len(user_items)
    user_items = new_user_items
    if len(user_items) == prev_len:
        break

# Recompute valid_items from final user_items
valid_items = set()
for uid, items in user_items.items():
    for iid, ts in items:
        valid_items.add(iid)

print(f"After filtering - Users: {len(user_items)}, Items: {len(valid_items)}")
print(f"Total interactions: {sum(len(v) for v in user_items.values())}")

# ============================================================
# Step 3: Remap user and item IDs to consecutive 0-based
# ============================================================
print("\nRemapping IDs ...")
sorted_users = sorted(user_items.keys())
sorted_items = sorted(valid_items)

user_id_map = {org_id: remap_id for remap_id, org_id in enumerate(sorted_users)}
item_id_map = {org_id: remap_id for remap_id, org_id in enumerate(sorted_items)}

num_users = len(user_id_map)
num_items = len(item_id_map)
print(f"Remapped users: {num_users}, items: {num_items}")

# ============================================================
# Step 4: Split into train/test (leave-last-one-out by timestamp)
# ============================================================
print("\nSplitting train/test ...")
train_data = defaultdict(list)
test_data = defaultdict(list)

user_seq_train = {}  # remap_uid -> [remap_iid, ...] 训练序列（不含test item）
user_seq_test = {}   # remap_uid -> remap_iid  测试item

for org_uid, items in user_items.items():
    remap_uid = user_id_map[org_uid]
    items_sorted = sorted(items, key=lambda x: x[1])
    # train序列: 去掉最后一个（避免数据泄露）
    user_seq_train[remap_uid] = [item_id_map[iid] for iid, ts in items_sorted[:-1]]
    # test: 最后一个
    last_iid, last_ts = items_sorted[-1]
    user_seq_test[remap_uid] = item_id_map[last_iid]
    # 同时填充 train_data / test_data
    for iid, ts in items_sorted[:-1]:
        train_data[remap_uid].append(item_id_map[iid])
    test_data[remap_uid].append(item_id_map[last_iid])

# ============================================================
# Step 4.5: Save user_seq_train.pkl & user_seq_test.pkl
# ============================================================
print("Writing user_seq_train.pkl & user_seq_test.pkl ...")
with open("user_seq_train.pkl", "wb") as f:
    pickle.dump(user_seq_train, f)
with open("user_seq_test.pkl", "wb") as f:
    pickle.dump(user_seq_test, f)
print(f"  train seq: {len(user_seq_train)} users, avg len = {np.mean([len(v) for v in user_seq_train.values()]):.1f}")
print(f"  test: {len(user_seq_test)} users, 1 item each")

# ============================================================
# Step 5: Write train.txt and test.txt
# ============================================================
print("Writing train.txt and test.txt ...")

with open("train.txt", "w") as f:
    for uid in range(num_users):
        items = train_data.get(uid, [])
        line = str(uid) + " " + " ".join(str(i) for i in items)
        f.write(line + "\n")

with open("test.txt", "w") as f:
    for uid in range(num_users):
        items = test_data.get(uid, [])
        line = str(uid) + " " + " ".join(str(i) for i in items)
        f.write(line + "\n")

# ============================================================
# Step 6: Write user_list.txt and item_list.txt
# ============================================================
print("Writing user_list.txt and item_list.txt ...")

with open("user_list.txt", "w") as f:
    f.write("org_id remap_id\n")
    for org_id, remap_id in sorted(user_id_map.items(), key=lambda x: x[1]):
        f.write(f"{org_id} {remap_id}\n")

with open("item_list.txt", "w") as f:
    f.write("org_id remap_id\n")
    for org_id, remap_id in sorted(item_id_map.items(), key=lambda x: x[1]):
        f.write(f"{org_id} {remap_id}\n")

# ============================================================
# Step 7: Generate trnMat.pkl and tstMat.pkl
# ============================================================
print("Generating trnMat.pkl and tstMat.pkl ...")

row_list, col_list, data_list = [], [], []
for uid in range(num_users):
    for iid in train_data.get(uid, []):
        row_list.append(uid)
        col_list.append(iid)
        data_list.append(1)
trnMat = coo_matrix((np.array(data_list), (np.array(row_list), np.array(col_list))), shape=(num_users, num_items))
with open("trnMat.pkl", "wb") as f:
    pickle.dump(trnMat, f)

row_list, col_list, data_list = [], [], []
for uid in range(num_users):
    for iid in test_data.get(uid, []):
        row_list.append(uid)
        col_list.append(iid)
        data_list.append(1)
tstMat = coo_matrix((np.array(data_list), (np.array(row_list), np.array(col_list))), shape=(num_users, num_items))
with open("tstMat.pkl", "wb") as f:
    pickle.dump(tstMat, f)

print("\n=== Summary ===")
print(f"Users: {num_users}")
print(f"Items: {num_items}")
print(f"Train interactions: {trnMat.nnz}")
print(f"Test interactions: {tstMat.nnz}")
print(f"Density: {(trnMat.nnz + tstMat.nnz) / (num_users * num_items) * 100:.4f}%")
print("Done! Files: train.txt, test.txt, user_list.txt, item_list.txt, trnMat.pkl, tstMat.pkl")
