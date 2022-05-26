function myfun(src,event)
    global index
    global list
    global dirpath
    global t
    global sig_norm
    global sig
    global index_ppg
    global index_abp
    
    if(event.Key=='return')
        time_pulse_waves_ppg = [];
        time_pulse_waves_abp = [];
        pulse_waves_ppg = [];
        pulse_waves_abp = [];
        
        h_fig = figure(1);
        axObjs = h_fig.Children;
        dataObjs = axObjs.Children;

        %% DECOUPAGE DES ZONES SUPPRIMEES PAR l'UTILISATEUR
        [end_points] = find(isnan(dataObjs(1).YData(2:end)) & not(isnan(dataObjs(1).YData(1:end-1))));
        [start_points] = find(isnan(dataObjs(1).YData(1:end-1)) & not(isnan(dataObjs(1).YData(2:end))))+1;

        start_points = [1 start_points];
        end_points = [end_points length(dataObjs(1).YData)];
        
        for i=1:length(start_points)
            %% DETECTION DES MIN SUR LES ONDES ABP ET PPG
            t2 = t(start_points(i):end_points(i));
            s_ppg = sig(index_ppg, start_points(i):end_points(i));
            s_abp = sig(index_abp, start_points(i):end_points(i));
            
            s_ppg_norm = sig_norm(index_ppg, start_points(i):end_points(i));
            s_abp_norm = sig_norm(index_abp, start_points(i):end_points(i));
            
            [locs_min_ppg, ~] = min_max_extraction_from_long_signals(t2, s_ppg_norm', 8);
            [locs_min_abp, ~] = min_max_extraction_from_long_signals(t2, s_abp_norm', 8);
            hold off
            
            %% Suppression des min en trop
            if (length(locs_min_ppg) ~= length(locs_min_abp))
                answer = questdlg('La taille des min entre la PPG et l''ABP est différente. Supprimer le(s) min(s) en trop au début ou à la fin ?', 'Conflit de taille', 'Début','Fin','Fin');
                d = length(locs_min_ppg) - length(locs_min_abp);
                
                if (strcmp(answer, 'Fin'))
                    if d>0
                        locs_min_ppg(end-d+1) = [];
                    else
                        locs_min_abp(end-d-1) = [];
                    end
                else
                    if d>0
                        locs_min_ppg(1:d) = [];
                    else
                        locs_min_abp(1:-d) = [];
                    end
                end
            end
            
            %% EXTRACTION DES ONDES
            [time_pulse_waves_ppg_temp, pulse_waves_ppg_temp] = pulse_waves_extraction_and_interpolation(t2, s_ppg, locs_min_ppg, 256);
            [time_pulse_waves_abp_temp, pulse_waves_abp_temp] = pulse_waves_extraction_and_interpolation(t2, s_abp, locs_min_abp, 256);
            
            time_pulse_waves_ppg = [time_pulse_waves_ppg time_pulse_waves_ppg_temp];
            time_pulse_waves_abp = [time_pulse_waves_abp time_pulse_waves_abp_temp];

            pulse_waves_ppg = [pulse_waves_ppg pulse_waves_ppg_temp];
            pulse_waves_abp = [pulse_waves_abp pulse_waves_abp_temp];
        end
        
        %% AFFICHAGE DES ONDES SEGMENTEES ET SAUVEGARDE
        figure(3)
        subplot(2,1,1)
        plot(time_pulse_waves_ppg, pulse_waves_ppg)
        title('PPG')
        
        subplot(2,1,2)
        plot(time_pulse_waves_abp, pulse_waves_abp)
        title('ABP')
        
        waitforbuttonpress
        
        answer = questdlg('Sauvegarder ?', 'Save', 'Oui','Non','Oui');
        if (strcmp(answer, 'Oui'))
            save([dirpath '\' list(index).name(1:end-5) '_pulses.mat'], 'time_pulse_waves_ppg', 'pulse_waves_ppg', 'time_pulse_waves_abp', 'pulse_waves_abp')
            index = index + 2;
        else
            answer = questdlg('Recommencer ?', 'Restart', 'Oui','Non','Oui');
            if (strcmp(answer, 'Non'))
                index = index + 2;
            end
        end
        
        if (index<length(list))
            figure(2)
            close
            figure(3)
            close
            display()
        end
    end
end