% show the pointclouds
clc;clear;close all;
vis = 0;
radius = 0.05;
evaluate_results_root = '../results/my_single_scale_normal_with_mask_res_models_0.5_0.5_multi2_2100/';
data_root = '../pclouds/';
% testlist = 'testset_vardensity_striped.txt';
testlist = 'testset_no_noise.txt';

test_name = importdata([data_root testlist]);
Total = 0;
Total_pgp10 = 0;
Total_pgp5 = 0;
count= 0;
for i=1:length(test_name)
    count=count+1;
    disp(count);
% 
%     if i ==1 ||i==3||i==10||i==14||i==15
%         continue;
%     end

    current_name = test_name{i};
    disp(current_name);
    points_gt_name = [data_root current_name '.xyz'];
    normals_gt_name = [data_root current_name '.normals'];
    pidx_gt_name = [data_root current_name '.pidx'];
    normals_result_name1 = [evaluate_results_root current_name '_scale1.normals'];
    normals_result_name2 = [evaluate_results_root current_name '_scale2.normals'];
    normals_result_name3 = [evaluate_results_root current_name '_scale3.normals'];

    max_VVV = [evaluate_results_root current_name '_v.normals'];
    points_gt = load(points_gt_name);
    bbdiag = norm(max(points_gt) -min(points_gt), 2);
    abs_radius = radius*bbdiag;
    normals_gt = load(normals_gt_name);
    pidx_gt = load(pidx_gt_name);
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

    mav_v_v = load(max_VVV);
    [aaa, max_indd] = max(mav_v_v,[],2);
    for iii = 1:100000
        if max_indd(iii)==1
            normals_result(iii,:) = normals_result1(iii,:);

        else if max_indd(iii)==2
              normals_result(iii,:) = normals_result2(iii,:);

            else
                normals_result(iii,:) = normals_result3(iii,:);
                
            end
        end
    end
    %% distance error:
%     normals_gt = normals_gt./sqrt(sum((normals_gt).^2,2));
%     normals_result = normals_result./sqrt(sum((normals_result).^2,2));
%     nn = sum(normals_gt.*normals_result,2);
%     nn(nn>1)=1;
%     nn(nn<-1)=1;
%     ang = rad2deg(acos(abs(nn)));
%     ang_o = rad2deg(acos(nn));
    diff = abs(sum(normals_result(pidx_gt,:).*normals_gt(pidx_gt,:),2))./ (sqrt(sum(normals_result(pidx_gt,:).^2,2)).* sqrt(sum(normals_gt(pidx_gt,:).^2,2)));
    diff(diff > 1) = 1;
    ang = acosd(diff);
    
    temp_ang = 0.5*ones(length(ang),1);
    temp_ang(ang>5)=0;
    temp_ang(ang>10)=1;
    
    temp_ang10 = ang;
    temp_ang10(ang<10)=1;
    temp_ang10(ang>=10)=0;
    pgp10 = sum(temp_ang10)/length(temp_ang10);
    temp_ang5 = ang;
    temp_ang5(ang<5)=1;
    temp_ang5(ang>=5)=0;
    pgp5 = sum(temp_ang5)/length(temp_ang5);
%     rms_ang  = mean(ang);
    rms_ang = sqrt(mean(ang.^2));
%     rms_ang_o = sqrt(mean(ang_o.^2));
%
    error1 = sqrt(sum((normals_gt-normals_result).^2,2));
    error2 = sqrt(sum((normals_gt+normals_result).^2,2));
    error = min(error1,error2);
    E = mean(error);
    Total= Total+rms_ang;
    Total_pgp10 = Total_pgp10 + pgp10;
    Total_pgp5 = Total_pgp5 + pgp5;
    disp(pgp5)
    disp(pgp10)
    disp(rms_ang);
    %% angle error:
%      error1 = abs(angle(acos(dot(normals_gt,normals_result,2)))*180/pi);
%      error2 =  abs(angle(acos(dot(normals_gt,-normals_result,2)))*180/pi);
%     error = min(error1,error2);
    if vis==1
%         s = ones(size(error));
%         %scatter3(points_gt(:,1),points_gt(:,2),points_gt(:,3),s, error,'filled');
        fig_h = figure('color',[1 1 1]);
        ax_h = axes('position',[0, 0, 1, 1]);
%         set_view(ax_h);
        set_vis_props(fig_h, ax_h);
%         colormap('parula');
%         caxis([0, 60]);
%         error_ang = ang;
% %         scatter3(points_gt(:,1),points_gt(:,2),points_gt(:,3),s, index/3,'filled');
%         scatter3(points_gt(:,1),points_gt(:,2),points_gt(:,3),s, error_ang,'filled');
% %         hold on;
% %         scatter3(points_gt(pidx_gt,1),points_gt(pidx_gt,2),points_gt(pidx_gt,3),'filled');
% 
%         axis equal;
%         axis off;
        point_size = 100;
        pc_h = scatter3(points_gt(:, 1), points_gt(:, 2), points_gt(:, 3), point_size, '.');
        axis off

        colormap('jet');
        caxis([0, 1]);
        pc_h.CData = temp_ang;
        active_axes = gca;
        active_axes.Position =  [0, 0, 1, 0.9];
        ax_rns_text = axes('position',[0, 0.9, 1, 0.1]);
        axis off
        text('string',num2str(rms_ang), 'position',[0.5, 0.5], 'FontSize', 24, 'HorizontalAlignment', 'center')
        image_filename = ['./', current_name, '.png'];    
        print(image_filename, '-dpng')
        gcf.CurrentAxes  = active_axes;
        active_axes.Position =  [0, 0, 1, 1];
        delete(ax_rns_text)
        
        

    end


end
disp(Total/length(test_name));
disp(Total_pgp5/(length(test_name)));
disp(Total_pgp10/(length(test_name)));
