from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, PathPatch
from matplotlib.transforms import Affine2D
import numpy as np
from matplotlib.cbook import get_sample_data
import mpl_toolkits.mplot3d.art3d as art3d

import cv2

from pathlib import Path
import sys

import pickle

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add code to path

path = str(FILE.parents[0])


def draw_point_3D(real_ball_trajectory_list, estimation_ball_trajectory_list, label = False):
        
        real_x = []
        real_y = []
        real_z = []
        esti_x = []
        esti_y = []
        esti_z = []



        for i in range((len(real_ball_trajectory_list))):

            real_x.append(real_ball_trajectory_list[i][0])
            real_y.append(real_ball_trajectory_list[i][1])
            real_z.append(real_ball_trajectory_list[i][2])

            ax.plot(real_x, real_y, real_z, c= 'red', zorder = 100)

        ax.scatter(real_x[0], real_y[0], real_z[0],s = 180, c='#FF3333', zorder = 101, marker = '*')
        


        for j in range((len(estimation_ball_trajectory_list))):

            esti_x.append(estimation_ball_trajectory_list[j][0])
            esti_y.append(estimation_ball_trajectory_list[j][1])
            esti_z.append(estimation_ball_trajectory_list[j][2])

            ax.plot(esti_x, esti_y, esti_z, '#3336FF', zorder = 100)

        ax.scatter(esti_x[0], esti_y[0], esti_z[0],s = 180, c='#3336FF', zorder = 101, marker = '*')


        if label == True:
            ax.plot(real_x[-1], real_y[-1], real_z[-1], 'red', zorder = 100, label = 'Actual trajectory')
            ax.plot(esti_x[-1], esti_y[-1], esti_z[-1], '#3336FF', zorder = 100, label = 'Predict trajectory')



#궤적
#real_ball_trajectory_list_5 = [[0,0,0]]
#estimation_ball_trajectory_list_5 =  [[-8.075992233062022, -2.0712591029119727, 2.0143476038492176], [-7.041614424760605, -1.9654479963268496, 1.998508488845969], [-6.065044696824322, -1.879832764271466, 1.9635872373067076], [-5.107183755941063, -1.7968180783990304, 1.925614986159784], [-4.169832727705863, -1.6838107370497974, 1.884759164010521], [-3.2611958072772738, -1.5644043096331428, 1.81101188769961], [-2.368649316849956, -1.4805807124989778, 1.737028408823873], [-1.4834200661316506, -1.378965700223568, 1.6496367839315553], [-0.7006541039923906, -1.2862463955187806, 1.5580097756344515]]

ball_trajectory =  [[-2.2092435836792, -1.5208747965392155, 1.8354100995074563], [-1.256748008728028, -1.2158820107576946, 1.7773671872057393], [-0.43730239868164134, -0.9512572790516179, 1.7647362439434577], [0.31844730377197195, -0.6813783702643058, 1.6464358897379312], [0.9937688827514641, -0.4760653461827718, 1.5745616247033452], [1.625661087036132, -0.2314306683633415, 1.4782900479509058], [2.253456306457519, 0.043343332772139115, 1.361194439767826], [2.89897174835205, 0.28222227616559953, 1.2284656521340567], [3.5127641677856447, 0.5366443786781522, 1.1123209164675203], [3.5127641677856447, 0.5116841750187032, 1.1123209164675203], [4.140768241882324, 0.7474247208730833, 0.9655034744212423], [4.769901943206787, 0.97797096615911, 0.7791678463511688], [5.398135852813721, 1.2108431143742158, 0.6090978922098989], [6.026425075531006, 1.428266203158699, 0.4098780358206572], [6.026425075531006, 1.428266203158699, 0.3927730513516908], [6.653767776489258, 1.6431816490162447, 0.18976757398738164], [7.318938446044922, 1.8357540816222784, 0.13111995424995382], [7.80655426979065, 2.0140895229051132, 0.30170535165753953], [8.336106252670287, 2.1801285433139634, 0.4512601625672338], [8.336106252670287, 2.1801285433139634, 0.4512601625672338], [8.878784608840942, 2.3222415406587897, 0.6027205829913534]]
real_ball_trajectory =  [[-1.9758686990321173, -1.2743506037066863, 1.743129350877194], [-1.1939511743124973, -1.0285721556308347, 1.7041339473684218], [-0.21340591689801117, -0.7027399213070984, 1.6318253578947373], [0.5452724074567715, -0.43403414550062563, 1.5517493754385967], [1.1673033846255532, -0.20216285373490195, 1.4655540526315796], [1.7802685723206535, 0.028046464346611122, 1.3647554736842116], [2.3775167039210077, 0.25235297940039314, 1.2522040736842122], [2.974764835521362, 0.4766594944541752, 1.1255014736842133], [3.5091447427427314, 0.6773547973970312, 1.000140789473688], [3.9806564255851162, 0.8544388882289627, 0.8801204210526365], [4.515036332806486, 1.0551341911718186, 0.7334349368421111], [5.002265071743617, 1.2381210850314812, 0.5898187894736908], [5.473776754586002, 1.4152051758634128, 0.44186842105263857], [5.992439605712625, 1.6099976757785375, 0.268935915789481], [6.511102456839248, 1.8047901756936622, 0.08533121052632302], [7.001520315932834, 1.9830897712965074, 0.03261101814792514], [7.422378872783627, 2.127799498535443, 0.17305217188241245], [7.856813512113478, 2.2771772814917646, 0.3081452854147862], [8.318400316401423, 2.435891175882856, 0.44068611854293327], [8.752834955731245, 2.5852689588391775, 0.555081632075307], [9.214421760019182, 2.743982853230269, 0.6656312652034545]]


plt.rcParams["figure.autolayout"] = True
fig_3d = plt.figure(figsize=(8,8),dpi=100)
ax = fig_3d.add_subplot(projection='3d')

ax.set_xlim(-12, 12)
ax.set_ylim(-6, 6)
ax.set_zlim(0,5)
ax.set_box_aspect((2, 1, 0.5))

img = cv2.imread(path + "/images/tennis_court_2.png")
img = cv2.resize(img, dsize=(1000,400), interpolation=cv2.INTER_LINEAR)

# cv2.imshow("img",img)
# cv2.waitKey(0)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
img=cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
img[:,:,3] = float(1)

stepX, stepY = 24 / img.shape[1], 12 / img.shape[0]

X1 = np.arange(-12, 12, stepX)
Y1 = np.arange(-6, 6, stepY)
#Y1 = np.arange(-8.4, 8.4, stepY)


X1, Y1 = np.meshgrid(X1, Y1)



ax.plot_surface(X1, Y1, np.ones(X1.shape) * -0.01,rstride=8, cstride=8, facecolors=img, zorder = 20)



#draw_point_3D(real_ball_trajectory_list_1, estimation_ball_trajectory_list_1)

#draw_point_3D(real_ball_trajectory_list_2, estimation_ball_trajectory_list_2)

#draw_point_3D(real_ball_trajectory_list_3, estimation_ball_trajectory_list_3)

#draw_point_3D(real_ball_trajectory_list_4, estimation_ball_trajectory_list_4)

#draw_point_3D(real_ball_trajectory_list_5, estimation_ball_trajectory_list_5,True)
draw_point_3D(ball_trajectory, real_ball_trajectory,True)



#낙하지점
"""draw_point_3D(real_ball_trajectory_list, estimation_ball_trajectory_list)

p = Ellipse((landing_point[0], landing_point[1]),  height = circle_radius_y,  width= circle_radius_x, angle = 0, color = "blue", zorder = 21 )
ax.add_patch(p)
art3d.pathpatch_2d_to_3d(p, z=0.01, zdir="z")"""






ax.view_init(30, 45)

ax.legend()

plt.show()