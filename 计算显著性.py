import netCDF4 as nc
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
#打开 NC 文件
Active=nc.Dataset('C:/Users/admin/Desktop/SMAP_L4_NC/Active.nc', 'r', format = 'NETCDF4')
Passive=nc.Dataset('C:/Users/admin/Desktop/SMAP_L4_NC/Passive.nc', 'r', format = 'NETCDF4')
SMAP_L4=nc.Dataset('C:/Users/admin/Desktop/SMAP_L4_NC/SMAP_L4.nc', 'r', format = 'NETCDF4')

sm_Active=np.asarray(Active.variables['sm'])
sm_Passive=np.asarray(Passive.variables['sm'])
sm_SMAP_L4=np.asarray(SMAP_L4.variables['sm'])

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

            T, AP_P = stats.ttest_ind(Active_Array, Passive_Array)
            AP_P_matrix.append(AP_P)
            T, AS_P = stats.ttest_ind(Active_Array, SMAP_L4_Array)
            AS_P_matrix.append(AS_P)
            T, PS_P = stats.ttest_ind(SMAP_L4_Array, Passive_Array)
            PS_P_matrix.append(PS_P)

AP_P_matrix=np.array(AP_P_matrix)
coff_AP=np.array(AP_P_matrix).reshape(9,12)

AS_P_matrix=np.array(AS_P_matrix)
coff_AS=np.array(AS_P_matrix).reshape(9,12)

PS_P_matrix=np.array(PS_P_matrix)
coff_PS=np.array(PS_P_matrix).reshape(9,12)

# plt.figure(figsize=(10,8))
# sns.heatmap(coff_AP,annot=True,cmap="YlGnBu")
# plt.title('Pearson_AP', fontsize=14)
# plt.savefig('C:/Users/Yuwan/Desktop/NC_2020/Pic/AP显著性.jpg')
# plt.show()

A_APS_matrix=np.array(A_APS_matrix).reshape(9,12)
P_APS_matrix=np.array(P_APS_matrix).reshape(9,12)
S_APS_matrix=np.array(S_APS_matrix).reshape(9,12)

matrix=[]
x_count=9
y_count=12
for i in range(x_count):
    for j in range(y_count):
        if 0<coff_AP[i][j]<=0.05 and 0<coff_AS[i][j]<=0.05 and 0<coff_PS[i][j]<=0.05:
            matrix.append('Average')
        elif coff_AP[i][j]>0.05 and 0<coff_AS[i][j]<=0.05 and 0<coff_PS[i][j]<=0.05:
            matrix.append('S')
        elif 0<coff_AP[i][j]<=0.05 and 0< coff_AS[i][j] <= 0.05 and coff_PS[i][j] > 0.05:
            matrix.append('A')
        elif 0<coff_AP[i][j]<=0.05 and coff_AS[i][j] > 0.05 and coff_PS[i][j] <= 0.05:
            matrix.append('P')
        elif coff_AP[i][j] > 0.05 and coff_AS[i][j] > 0.05 and 0<coff_PS[i][j] <= 0.05:
            matrix.append('PS')
        elif coff_AP[i][j] > 0.05 and 0<coff_AS[i][j] <= 0.05 and coff_PS[i][j] > 0.05:
            matrix.append('AS')
        elif 0<coff_AP[i][j] <= 0.05 and coff_AS[i][j] > 0.05 and coff_PS[i][j] > 0.05:
            matrix.append('AP')
        elif coff_AP[i][j] > 0.05 and coff_AS[i][j] > 0.05 and coff_PS[i][j] > 0.05:
            matrix.append('S')
        elif coff_AP[i][j] <= 0 or coff_AS[i][j] <= 0 or coff_PS[i][j] <= 0:
            matrix.append('S')
        else:
            matrix.append('None')
matrix=np.array(matrix).reshape(x_count,y_count)
# print(matrix)

pixel=[]
for a in range(366):
  for b in range(9):
     for c in range(12):
         if sm_Passive[a][b][c]==0:
             pixel_None=sm_SMAP_L4[a][b][c]
             pixel.append(pixel_None)
         else:
            if matrix[b][c]=='Average':
                pixel_Ave=A_APS_matrix[b][c]/(A_APS_matrix[b][c]+P_APS_matrix[b][c]+S_APS_matrix[b][c])*sm_Active[a][b][c]+P_APS_matrix[b][c] / (A_APS_matrix[b][c]+P_APS_matrix[b][c]+S_APS_matrix[b][c]) * sm_Passive[a][b][c]+S_APS_matrix[b][c] / (A_APS_matrix[b][c]+ P_APS_matrix[b][c]+ S_APS_matrix[b][c]) * sm_SMAP_L4[a][b][c]
                pixel.append(pixel_Ave)
            elif matrix[b][c]=='S':
                pixel_S=sm_SMAP_L4[a][b][c]
                pixel.append(pixel_S)
            elif matrix[b][c]=='A':
                pixel_A= sm_Active[a][b][c]
                pixel.append(pixel_A)
            elif matrix[b][c]=='P':
                pixel_P= sm_Passive[a][b][c]
                pixel.append(pixel_P)
            elif matrix[b][c]=='PS':
                pixel_PS= (sm_SMAP_L4[a][b][c]+sm_Passive[a][b][c])/2
                pixel.append(pixel_PS)
            elif matrix[b][c]=='AP':
                pixel_AP=(sm_Active[a][b][c]+sm_Passive[a][b][c])/2
                pixel.append(pixel_AP)
            elif matrix[b][c]=='AS':
                pixel_AS=(sm_Active[a][b][c]+sm_SMAP_L4[a][b][c])/2
                pixel.append(pixel_AS)
            else:
                pixel_0= 0
                pixel.append(pixel_0)

pixel_Arr=np.array(pixel).reshape(366,9,12) #融合结果

sm_Active_chw = np.transpose(sm_Active, (1, 2, 0))
sm_Passive_chw = np.transpose(sm_Passive, (1, 2, 0))
sm_SMAP_L4_chw = np.transpose(sm_SMAP_L4, (1, 2, 0))
Merge=np.transpose(pixel_Arr, (1, 2, 0))

df = pd.read_csv('C:/Users/admin/Desktop/SMAP_L4_NC/SMC.csv')
SMC = df['43'][1:367]
#
x=np.arange(1,367,1)
y_Active=sm_Active_chw[4][4]
y_Passive=sm_Passive_chw[4][4]
y_SMAP_L4=sm_SMAP_L4_chw[4][4]
y_Merge=Merge[4][4]
y_SMC=np.array(SMC)

sta_Active=np.array(y_Active)
sta_Passive=np.array(y_Passive)
sta_SMAP_L4=np.array(y_SMAP_L4)
sta_Merge=np.array(y_Merge)
sta_SMC=np.array(y_SMC)


mse_Active = np.sum((sta_Active - sta_SMC) ** 2) / len(sta_Active)
rmse_Active = sqrt(mse_Active)
r2_Active = 1-mse_Active/ np.var(sta_SMC)

mse_Passive = np.sum((sta_Passive - sta_SMC) ** 2) / len(sta_Passive)
rms_Passive = sqrt(mse_Passive)
r2_Passive = 1-mse_Passive/ np.var(sta_SMC)

mse_SMAP_L4 = np.sum((sta_SMAP_L4 - sta_SMC) ** 2) / len(sta_SMAP_L4)
rmse_SMAP_L4 = sqrt(mse_SMAP_L4)
r2_SMAP_L4 = 1-mse_SMAP_L4/ np.var(sta_SMC)

mse_Merge = np.sum((sta_Merge - sta_SMC) ** 2) / len(sta_Merge)
rmse_Merge = sqrt(mse_Merge)
r2_Merge = 1-mse_Merge/ np.var(sta_SMC)

print('rmse_Active_4_4=',rmse_Active,'rmse_Passive_4_4=',rms_Passive,'rmse_SMAP_L4_4_4=',rmse_SMAP_L4,'rmse_Merge_4_4=',rmse_Merge)
print('r2_Active_4_4=',r2_Active,'r2_Passive_4_4=',r2_Passive,'r2_SMAP_L4_4_4=',r2_SMAP_L4,'r2_Merge_4_4=',r2_Merge)

#
plt.scatter(x, y_Active,label = 'Active (rmse=0.063 r2=0.27)',color='#F1D77E')
plt.scatter(x, y_Passive,label = 'Passive (rmse=0.064 r2=0.25)',color='#B1CE46')
plt.scatter(x, y_SMAP_L4,label = 'SMAP_L4 (rmse=0.064 r2=0.27)',color='#9DC3E7')
plt.plot(x, y_Merge,label = 'Merge (rmse=0.052 r2=0.51)',color='#9394E7')
plt.plot(x, y_SMC,label = 'IoT',color='#D76364')
plt.xlabel('DOY')
plt.ylabel('SMC')
plt.legend(loc='upper left')
plt.savefig('C:/Users/admin/Desktop/SMAP_L4_NC/SMC.png')
plt.show()

# 保存最终的SM
input_NCdata=np.array(pixel_Arr) #输入数据
lon=np.asarray(SMAP_L4.variables['lon']) # refer
lat=np.asarray(SMAP_L4.variables['lat']) # refer
ncfile = nc.Dataset('C:/Users/admin/Desktop/SM_m/SM_25km.nc' ,'w' ,format = 'NETCDF4') #保存NC路径

# 添加坐标轴（经度纬度和时间）
xdim = ncfile.createDimension('lon' ,12) #300
ydim = ncfile.createDimension('lat' ,9) #225
tdim = ncfile.createDimension('time',366)

# # 添加全局属性，比如经纬度和标题，主要是对数据进行一个简单的介绍
ncfile.setncattr_string('title' ,'TEMPERATURE')
ncfile.setncattr_string('geospatial_lat_min' ,'-36.476 degrees')
ncfile.setncattr_string('geospatial_lat_max' ,'-34.476 degrees')
ncfile.setncattr_string('geospatial_lon_min' ,'146.655 degrees')
ncfile.setncattr_string('geospatial_lon_max' ,'149.405 degrees')
#
# # 添加变量和局部属性，存入数据
var = ncfile.createVariable(varname='lon' ,datatype=np.float64,dimensions='lon')
var.setncattr_string('long_name' ,'longitude')
var.setncattr_string('units' ,'degrees_east')
var[: ] =lon
#
var = ncfile.createVariable(varname='lat' ,datatype=np.float64 ,dimensions='lat')
var.setncattr_string('long_name' ,'latitude')
var.setncattr_string('units' ,'degrees_north')
var[: ] =lat
#
tvar = ncfile.createVariable(varname='time', datatype=np.float64 ,dimensions='time')
tvar.setncattr_string('long_name' ,'time')
tvar.setncattr_string('units' ,'days since 0000-01-01')
tvar.calendar = "standard"
tvar[: ] =366
#
var = ncfile.createVariable(varname='SM' ,datatype=np.float64 ,dimensions=('time' ,'lat' ,'lon'))
var.setncattr_string('long_name' ,'SM')
var.setncattr_string('units' ,'%')
var[: ] =input_NCdata

# 关闭文件
ncfile.close()
print('finished')