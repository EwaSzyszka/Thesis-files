list_of_centroids = [(638, 293), (638, 290), (640, 289), (641, 290), (637, 289), (619, 321), (633, 285), (627, 300), (610, 286), (562, 298), (523, 297), (476, 289), (423, 295), (379, 309), (326, 314), (287, 307), (245, 298), (221, 299), (207, 297), (197, 299), (193, 300), (194, 300), (203, 300), (228, 289), (259, 300), (289, 301), (337, 301), (369, 301), (402, 288), (436, 296), (483, 284), (532, 303), (559, 306), (587, 303), (639, 272), (671, 273), (715, 272), (767, 271), (810, 279), (850, 278), (889, 278), (930, 280), (982, 285), (1008, 283), (1033, 279), (1032, 276), (1017, 281), (987, 283), (950, 290), (913, 289), (879, 291), (846, 302), (809, 303), (766, 310), (727, 304), (675, 311), (614, 317), (565, 336), (542, 338), (518, 332), (490, 323), (449, 319), (419, 318), (397, 316), (379, 320), (351, 318), (329, 317), (306, 326), (294, 332), (266, 322), (247, 313), (231, 306), (230, 297), (236, 295), (254, 312), (282, 309), (323, 303), (378, 304), (425, 302), (477, 298), (518, 304), (547, 310), (593, 297), (650, 277), (729, 274), (795, 308), (859, 304), (909, 313), (947, 326), (978, 330), (997, 331), (1011, 331), (1028, 333), (1031, 333), (1028, 338), (1027, 336), (1022, 336), (1020, 334), (1018, 330), (1014, 333), (1012, 333), (1011, 333), (1011, 332), (1007, 336), (1007, 336), (1002, 335), (994, 333), (991, 326), (978, 328), (973, 324), (973, 319), (970, 321), (961, 332), (953, 334), (934, 325), (934, 329), (923, 315), (925, 325), (918, 313), (920, 324)]

#ACCESSING THE FIRST COORDINATE - change 0 to 1 if you want to access the second coordinate
#list_of_centroids[i][0]  ----- x coordinates
#list_of_centroids[i][1]  ----- y coordinates

threshold_upper = 0.0
threshold_middle = 0.0
threshold_lower = 0.0

threshold_left = 0.0
threshold_middle_vertical = 0.0
threshold_right = 0.0

################## Checking to which HORIZONTAL tiers did the majority of the points fell ##################

for i in range(len(list_of_centroids)):

    #HORIZONTAL DETECTORS

    if list_of_centroids[i][0] in range(0,1222) and list_of_centroids[i][1] in range(0,250):
        threshold_upper+= 1

    if list_of_centroids[i][0] in range(0,1222) and list_of_centroids[i][1] in range(250,550):
        threshold_middle+= 1

    if list_of_centroids[i][0] in range(0,1222) and list_of_centroids[i][1] in range(550,720):
        threshold_lower+= 1

    #VERTICAL DETECTORS

    if list_of_centroids[i][0] in range(0,400) and list_of_centroids[i][1] in range(0,716):
        threshold_left+= 1

    if list_of_centroids[i][0] in range(400,800) and list_of_centroids[i][1] in range(0,716):
        threshold_middle_vertical+= 1

    if list_of_centroids[i][0] in range(800,1280) and list_of_centroids[i][1] in range(0,716):
        threshold_right+= 1


#80% threshold is set up
cap = 0.7

if (threshold_upper/len(list_of_centroids)) > cap:
    print('Upper horizontal movement')
elif (threshold_middle/len(list_of_centroids)) > cap:
    print('Middle horizontal movement')
elif (threshold_lower/len(list_of_centroids)) > cap:
    print('Lower horizontal movement')

elif (threshold_left/len(list_of_centroids)) > cap:
    print('Left vertical movement')
elif (threshold_middle_vertical/len(list_of_centroids)) > cap:
    print('Middle vertical movement')
elif (threshold_right/len(list_of_centroids)) > cap:
    print('Right vertical movement')

else:
    print('total chaos')



print(threshold_upper/len(list_of_centroids))
print(threshold_middle/len(list_of_centroids))
print(threshold_lower/len(list_of_centroids))
print((threshold_left/len(list_of_centroids)))
print((threshold_middle_vertical/len(list_of_centroids)))
print((threshold_right/len(list_of_centroids)))

#############################################################################################################

    #print(list_of_centroids[i][0]) #x coordinates

    #if cnt_centroid[0] in range(0,900) and cnt_centroid[1] in range(300,400):
#Dividing the screen into ranges:
