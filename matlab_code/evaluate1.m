function [error,rms_ang,ang] = evaluate(normals_gt,normals_result)
    
normals_gt = normals_gt./sqrt(sum((normals_gt).^2,2));
normals_result = normals_result./sqrt(sum((normals_result).^2,2));
nn = sum(normals_gt.*normals_result,2);
nn(nn>1)=1;
nn(nn<-1)=1;
ang = rad2deg(acos(abs(nn)));
ang_o = rad2deg(acos(nn));

rms_ang = mean(ang);
% rms_ang = sqrt(mean(ang.^2));

rms_ang_o = sqrt(mean(ang_o.^2));

error1 = sqrt(sum((normals_gt-normals_result).^2,2));
error2 = sqrt(sum((normals_gt+normals_result).^2,2));
error = min(error1,error2);
end