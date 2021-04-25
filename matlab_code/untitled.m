% show the pointclouds
clc;clear;close all;
vis = 1;
radius = 0.05;
evaluate_results_root1 = '../results/my_single_scale_normal_with_mask_res/';
% evaluate_results_root1 = '../results/my_single_scale_normal/';

evaluate_results_root2 = '../results/my_single_scale_normal_with_mask_res_models_0.5_0.5_r0.03/';
evaluate_results_root3 = '../results/my_single_scale_normal_with_mask_res_models_0.5_0.5_r0.01/';

% evaluate_results_root1 = '../results_train/my_single_scale_normal_with_mask_res_models_0.5_0.5_r0.05/';
% evaluate_results_root2 = '../results_train/my_single_scale_normal_with_mask_res_models_0.5_0.5_r0.03/';
% evaluate_results_root3 = '../results_train/my_single_scale_normal_with_mask_res_models_0.5_0.5_r0.01/';


data_root = '../pclouds/';
% testlist = 'trainingset_whitenoise.txt';
testlist = 'testset_no_noise.txt';

test_name = importdata([data_root testlist]);
Total = 0;
Total1= 0;
Total2 =0;
Total3 = 0;
count= 0;
for i=1:length(test_name)
    count=count+1;
    disp(count);
% 
%     if i ==1 ||i==2||i==10||i==14||i==15
%         continue;
%     end
    current_name = test_name{i};
    disp(current_name);
    points_gt_name = [data_root current_name '.xyz'];
    normals_gt_name = [data_root current_name '.normals'];
    pidx_gt_name = [data_root current_name '.pidx'];
    normals_result_name1 = [evaluate_results_root1 current_name '.normals'];
    normals_result_name2 = [evaluate_results_root2 current_name '.normals'];
    normals_result_name3 = [evaluate_results_root3 current_name '.normals'];
    points_gt = load(points_gt_name);
    normals_gt = load(normals_gt_name);

%     bbdiag = norm(max(points_gt) -min(points_gt), 2);
%     abs_radius = radius*bbdiag;
%     pidx_gt = load(pidx_gt_name);
%     for j=1:length(pidx_gt)
%         center = points_gt(pidx_gt(j),:);
%         center_normal = normals_gt(pidx_gt(j),:);
%         [idx,D] = rangesearch(points_gt,center,abs_radius);
%         patch = points_gt(idx{1:end},:);
%         patch_normal = normals_gt(idx{1:end},:);
%         delta1 = sqrt(sum((patch_normal-center_normal).^2,2));
%         delta2 = sqrt(sum((patch_normal+center_normal).^2,2));
%         delta = min(delta1,delta2);
%         delta = (delta-min(delta))/(max(delta)-min(delta));
%         delta(delta<0.5)=0;
%         delta(delta>=0.5)=1;
%         if vis==1
%             scatter3(center(:,1),center(:,2),center(:,3),'*');
%             hold on;
% %             scatter3(points_gt(:,1),points_gt(:,2),points_gt(:,3),'.');
% %             hold on;
%             s = 10*ones(size(delta));
%             scatter3(patch(:,1),patch(:,2),patch(:,3),s, delta,'filled');
%             axis equal;
%             axis off;
%         end
%     
%     end
    
    normals_result1 = load(normals_result_name1);
    normals_result2 = load(normals_result_name2);
    normals_result3 = load(normals_result_name3);

    %% distance error:
    
    [error1,rms_ang1,ang1] = evaluate(normals_gt,normals_result1);
    [error2,rms_ang2,ang2]= evaluate(normals_gt,normals_result2);
    [error3,rms_ang3,ang3] = evaluate(normals_gt,normals_result3);
    [rms_ang,index] = min([ang1 ang2 ang3],[],2);
    rms_ang = sqrt(mean(rms_ang.^2));
    
%     E = mean(error);
    Total1 = Total1 +rms_ang1;
    Total2 = Total2 +rms_ang2;
    Total3 = Total3 +rms_ang3;
    Total= Total+rms_ang;
    
%%  
    bbdiag = norm(max(points_gt) -min(points_gt), 2);
    D_t = [ang1 ang2 ang3];
    DD = [];
%     DD = zeros(size(points_gt,1),1);
    count=0;
    for iii =1:20:size(points_gt,1)
        count = count+1
        abs_radius = 0.01*bbdiag;
        
        center = points_gt(iii,:);
%         [idx,D] = rangesearch(points_gt,center,abs_radius);
        [idx,D] = knnsearch(points_gt,center,'K',64,'IncludeTies',true);
        gt1 = normals_result1(idx{1:end},:)./sqrt(sum((normals_result1(idx{1:end},:)).^2,2)+1e-10);
        gt2 = normals_result2(idx{1:end},:)./sqrt(sum((normals_result2(idx{1:end},:)).^2,2)+1e-10);
        gt3 = normals_result3(idx{1:end},:)./sqrt(sum((normals_result3(idx{1:end},:)).^2,2)+1e-10);
        A1 = mean(mean(abs(gt1*gt1')));
        A2 =  mean(mean(abs(gt2*gt2')));
        A3 =  mean(mean(abs(gt3*gt3')));
        AA = min([A1 A2 A3]);
        if AA>0.8
            [v,index] = max([A1,A2],[],2);
        else
            [v,index] = min([A2,A3],[],2);
            index = index+1;
        end
        DD(count) = D_t(iii,index);
        disp([D_t(iii,:) D_t(iii,index)] );
        
        rms_ang_s = sqrt(mean(DD.^2));
        disp([rms_ang1,rms_ang2,rms_ang3,rms_ang,rms_ang_s]);
    end
    rms_ang_s = sqrt(mean(DD.^2));
    disp([rms_ang1,rms_ang2,rms_ang3,rms_ang,rms_ang_s]);

    %% angle error:

    
    
    
    error_ang =ang3;
    error_ang(error_ang>60)=60;
    error_ang = error_ang/60;
    if vis==1
        s = ones(size(error1));
        figure('color',[1 1 1]);
        scatter3(points_gt(:,1),points_gt(:,2),points_gt(:,3),s, index/3,'filled');
%         scatter3(points_gt(:,1),points_gt(:,2),points_gt(:,3),s, error_ang,'filled');

%         hold on;
%         scatter3(points_gt(pidx_gt,1),points_gt(pidx_gt,2),points_gt(pidx_gt,3),'filled');
        view(3);
        camlight;
        axis equal;
        axis off;

    end


end
disp([Total1/length(test_name) Total2/length(test_name) Total3/length(test_name) Total/length(test_name)]);


