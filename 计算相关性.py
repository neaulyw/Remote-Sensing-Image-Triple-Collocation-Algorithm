import netCDF4 as nc
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

#打开 NC 文件
Active=nc.Dataset('C:/Users/admin/Desktop/SMAP_L4_NC/Active.nc', 'r', format = 'NETCDF4')
Passive=nc.Dataset('C:/Users/admin/Desktop/SMAP_L4_NC/Passive.nc', 'r', format = 'NETCDF4')
SMAP_L4=nc.Dataset('C:/Users/admin/Desktop/SMAP_L4_NC/SMAP_L4.nc', 'r', format = 'NETCDF4')

sm_Active=np.array(Active.variables['sm'])
sm_Passive=np.array(Passive.variables['sm'])
sm_SMAP_L4=np.array(SMAP_L4.variables['sm'])

sm_Active_chw = np.transpose(sm_Active, (1, 2, 0))
sm_Passive_chw = np.transpose(sm_Passive, (1, 2, 0))
sm_SMAP_L4_chw = np.transpose(sm_SMAP_L4, (1, 2, 0))

count_0=sm_Active_chw.shape[0]
count_1=sm_Active_chw.shape[1]
count_2=sm_Active_chw.shape[2]

A_APS_matrix = []
P_APS_matrix = []
S_APS_matrix = []

AP_P_matrix = []
AS_P_matrix = []
PS_P_matrix = []

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

            AP_P = np.corrcoef(Active_Array, Passive_Array)
            AP_P_matrix.append(AP_P)
            print(AP_P_matrix)
            AS_P = np.corrcoef(Active_Array, SMAP_L4_Array)
            AS_P_matrix.append(AS_P)
            PS_P = np.corrcoef(SMAP_L4_Array, Passive_Array)
            PS_P_matrix.append(PS_P)

AP_P_matrix=np.array(AP_P_matrix)
count_AP=AP_P_matrix.shape[0]
coff_AP=[]
for i in range(count_AP):
    ave=AP_P_matrix[i,0,1]
    coff_AP.append(ave)
coff_AP=np.array(coff_AP).reshape(9,12)
print(coff_AP[0,0])
#
# AS_P_matrix=np.array(AS_P_matrix)
# count_AS=AS_P_matrix.shape[0]
# coff_AS=[]
# for i in range(count_AS):
#     ave=AS_P_matrix[i,0,1]
#     coff_AS.append(ave)
# coff_AS=np.array(coff_AS).reshape(9,12)
#
# PS_P_matrix=np.array(PS_P_matrix)
# count_PS=PS_P_matrix.shape[0]
# coff_PS=[]
# for i in range(count_PS):
#     ave=PS_P_matrix[i,0,1]
#     coff_PS.append(ave)
# coff_PS=np.array(coff_PS).reshape(9,12)
#
# plt.figure(figsize=(10,8))
# sns.heatmap(coff_AP,annot=True,cmap="YlGnBu")
# plt.title('coff_AP', fontsize=14)
# # plt.savefig('C:/Users/Yuwan/Desktop/NC_2020/Pic/AP相关性.jpg')
# plt.show()