function event_list_f=Del_ReadError(event_list)
    N_all=length(event_list);
    window_lim=20;
    
    del_index_repeat=[];
    for ii=1:1:N_all-window_lim+1
        data_mean= mean(event_list((ii):ii+window_lim-1));
        if data_mean<0.15 || data_mean>0.85
            del_index_repeat=[del_index_repeat,ii:1:ii+window_lim-1];
        end
    end
    del_index=unique(del_index_repeat);
    event_list(del_index)=[];
    event_list_f=event_list;
end