import matplotlib.pyplot as plt

# # List of list of keypoints
# keypoints = [
#     [[0.11117468674977621, 0.37122279979564526, 1.2056138515472412], [0.11416683991750082, 0.35923190646701386, 2.5873148441314697], [0.10369431177775065, 0.3618965431495949, 1.6630399227142334], [0.11865506966908773, 0.35923190646701386, 0.4780352711677551], [0.08648942311604818, 0.3632288614908854, 0.980438232421875], [0.12613545258839926, 0.3938722398546007, 0.3853132128715515], [0.06928453445434571, 0.40453081484194153, 0.3129774034023285], [0.15231680870056152, 0.4298449198404948, 0.7426730990409851], [0.04235515197118123, 0.4671499181676794, 0.4215819835662842], [0.1784981409708659, 0.4671499181676794, 0.8380668759346008], [0.022158116102218628, 0.5191104465060764, 0.5475670695304871],[0.154560915629069, 0.591055749963831, 0.7061923742294312], [0.09995411237080892, 0.6203668665002894, 0.930764377117157], [0.15755306879679362, 0.6576718365704571, 0.7785588502883911], [0.0939698060353597, 0.7469373914930556, 0.6001151204109192]],
#     [[0.6967281341552735, 0.36572644269024884, 2.271639108657837], [0.7034604390462239, 0.35640903049045136, 2.926218032836914], [0.6967281341552735, 0.3550779837149161, 1.4080067873001099], [0.722909418741862, 0.36838856449833624, 2.0775673389434814], [0.6967281341552735, 0.35774007726598667, 0.36450502276420593], [0.7535790125528972, 0.4216310289171007, 0.28746283054351807], [0.7004683176676433, 0.4189689353660301, 0.30842912197113037], [0.7872407913208008, 0.4695492497196904, 0.7166759371757507], [0.6697987238566081, 0.4389348347981771, 0.5782390236854553], [0.802949587504069, 0.5414265385380498, 1.328368067741394], [0.6391291300455729, 0.42961739434136287, 0.3367505967617035], [0.7685398101806641, 0.5520750257703994, 0.2824374735355377], [0.7341300328572591, 0.5494129322193287, 0.29072874784469604], [0.773776117960612, 0.6692084418402777, 0.6967765688896179], [0.6922398249308268, 0.6146348741319444, 0.1711421012878418], [0.8111780166625977, 0.76504477041739, 0.4932821989059448], [0.684759521484375, 0.7078091656720197, 0.612127959728241]],
#     ]
# # Plot each set of keypoints with a different color
# for keypoint_set in keypoints:
#     x_vals = [kp[0] for kp in keypoint_set]
#     y_vals = [kp[1] for kp in keypoint_set]
#     plt.scatter(x_vals, y_vals, label=f"Set {keypoints.index(keypoint_set) + 1}")

# # Formatting
# plt.xlabel("X values")
# plt.ylabel("Y values")
# plt.title("Scatter Plot of Keypoints")
# plt.legend()
# plt.grid(True)

# # Show the plot
# plt.show()


# import matplotlib.pyplot as plt
# import ast
# import pandas as pd

# df = pd.read_csv("midpoints3.csv")
# keypoints = df["Keypoints left"].apply(ast.literal_eval)
# scores = df["Left score"]

# # Plot each set of keypoints with a different color
# for i in range(len(keypoints)):
#     for joint in keypoints[i]:
#         if(scores[i]) > 0.8:    
#             print(joint)
#             x_vals = joint[0]
#             y_vals = joint[1]
#             plt.scatter(x_vals, y_vals)

# # Formatting
# plt.xlabel("X values")
# plt.ylabel("Y values")
# plt.title("Scatter Plot of Keypoints")
# plt.legend()
# plt.grid(True)

# # Show the plot
# plt.show()

import matplotlib.pyplot as plt
import ast
import pandas as pd

df = pd.read_csv("midpoints1.csv")
keypoints = df["Left player left hip"].apply(ast.literal_eval)
scores = df["Left score"]

# Plot each set of keypoints with a different color
for i in range(len(keypoints)):
    if(scores[i]) > 0.8:    
        print(keypoints[i])
        x_vals = keypoints[i][0]
        y_vals = keypoints[i][1]
        plt.scatter(x_vals, y_vals)

# Formatting
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Scatter Plot of Keypoints")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
