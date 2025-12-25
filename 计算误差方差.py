import netCDF4 as nc
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

#打开 NC 文件
Active=nc.Dataset('C:/Users/Yuwan/Desktop/NC_2020/SMAP_L4_NC/SMAP_L4_NC/Active.nc', 'r', format = 'NETCDF4')
Passive=nc.Dataset('C:/Users/Yuwan/Desktop/NC_2020/SMAP_L4_NC/SMAP_L4_NC/Passive.nc', 'r', format = 'NETCDF4')
SMAP_L4=nc.Dataset('C:/Users/Yuwan/Desktop/NC_2020/SMAP_L4_NC/SMAP_L4_NC/SMAP_L4.nc', 'r', format = 'NETCDF4')

# 转为数组
sm_Active=np.asarray(Active.variables['sm'])
sm_Passive=np.asarray(Passive.variables['sm'])
sm_SMAP_L4=np.asarray(SMAP_L4.variables['sm'])

sm_Active_chw = np.transpose(sm_Active, (1, 2, 0))
sm_Passive_chw = np.transpose(sm_Passive, (1, 2, 0))
sm_SMAP_L4_chw = np.transpose(sm_SMAP_L4, (1, 2, 0))
#
count_0=sm_Active_chw.shape[0]
count_1=sm_Active_chw.shape[1]
count_2=sm_Active_chw.shape[2]
#

A_APS_matrix = []
P_APS_matrix = []
S_APS_matrix = []

AP_P_matrix = []
AS_P_matrix = []
PS_P_matrix = []
#
def tcol_snr(x, y, z, ref_ind=0):
    cov = np.cov(np.vstack((x, y, z)))
    ind = (0, 1, 2, 0, 1, 2)
    no_ref_ind = np.where(np.arange(3) != ref_ind)[0]
    err_var = np.array([cov[i, i] - (cov[i, ind[i + 1]] * cov[i, ind[i + 2]]) / cov[ind[i + 1], ind[i + 2]] for i in np.arange(3)])
    return err_var
#
# #计算方差误差
for i in range(count_0):
    for j in range(count_1):
            Active_Array = np.array(sm_Active_chw[i, j, :]).flatten()
            Passive_Array = np.array(sm_Passive_chw[i, j, :]).flatten()
            SMAP_L4_Array = np.array(sm_SMAP_L4_chw[i, j, :]).flatten()
            APS = tcol_snr(Active_Array, Passive_Array, SMAP_L4_Array, ref_ind=0)
            A_APS = APS[0]
            A_APS_matrix.append(A_APS)
            P_APS = APS[1]
            P_APS_matrix.append(P_APS)
            S_APS = APS[2]
            S_APS_matrix.append(S_APS)

A_APS_matrix=np.array(A_APS_matrix).reshape(9,12) #误差方差
P_APS_matrix=np.array(P_APS_matrix).reshape(9,12) #误差方差
S_APS_matrix=np.array(S_APS_matrix).reshape(9,12) #误差方差

AP_P_matrix=np.array(AP_P_matrix)
AS_P_matrix=np.array(AS_P_matrix)
PS_P_matrix=np.array(PS_P_matrix)

negative_A_APS = A_APS_matrix < 0
negative_P_APS = P_APS_matrix < 0
negative_S_APS = S_APS_matrix < 0

count_A_APS = np.sum(negative_A_APS)
count_P_APS = np.sum(negative_P_APS)
count_S_APS = np.sum(negative_S_APS)

print('A_APS=',count_A_APS, 'P_APS=',count_P_APS,'S_APS=', count_S_APS)

plt.figure(figsize=(10,8))
sns.heatmap(S_APS_matrix,annot=False,cmap="RdBu_r",linewidths=0.3,linecolor="grey")
plt.title('SMAP_L4', fontsize=14)
plt.savefig('C:/Users/Yuwan/Desktop/NC_2020/Pic/SMAP_L4误差方差.jpg')
plt.show()