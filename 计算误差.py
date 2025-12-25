import netCDF4 as nc
import numpy as np

#打开 NC 文件
Active=nc.Dataset('C:/Users/Yuwan/Desktop/NC_2020/SMAP_L4_NC/SMAP_L4_NC/Active.nc', 'r', format = 'NETCDF4')
Passive=nc.Dataset('C:/Users/Yuwan/Desktop/NC_2020/SMAP_L4_NC/SMAP_L4_NC/Passive.nc', 'r', format = 'NETCDF4')
ERA5_LAND=nc.Dataset('C:/Users/Yuwan/Desktop/NC_2020/SMAP_L4_NC/SMAP_L4_NC/ERA5_LAND.nc', 'r', format = 'NETCDF4')

# 转为数组
sm_Active=np.asarray(Active.variables["sm"])
sm_Passive=np.asarray(Passive.variables["sm"])
sm_ERA5_LAND=np.asarray(ERA5_LAND.variables["sm"])

# #计算误差
def tcol_error(x, y, z):
    e_x = np.sqrt(np.abs(np.mean((x - y) * (x - z))))
    e_y = np.sqrt(np.abs(np.mean((y - x) * (y - z))))
    e_z = np.sqrt(np.abs(np.mean((z - x) * (z - y))))
    return e_x, e_y, e_z

tcol_error=tcol_error(sm_Active, sm_Passive, sm_ERA5_LAND)
print('error_Active=',tcol_error[0],'error_Passive=',tcol_error[1],'error_ERA5_LAND=',tcol_error[2])