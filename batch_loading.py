 # for epoch in range(epochs):
    #     start_tr=0
    #     start_val=0
    #     a_ind=0
    #     for i in range(int(50000/inst_tr)):
    #         print('train instances gone:', inst_tr*(i+1))
    #         print('val instances gone:', inst_val*(i+1))
    #         print(start_tr, start_tr + inst_tr)
    #         train_dataset = TranslationDataset(train_en_file, train_de_file, en_tokenizer, de_tokenizer, enc_maxlength, dec_maxlength, start_tr, inst_tr)
    #         valid_dataset = TranslationDataset(valid_en_file, valid_de_file, en_tokenizer, de_tokenizer, enc_maxlength, dec_maxlength, start_val, inst_val)
    #         start_tr=start_tr+inst_tr
    #         start_val=start_val+inst_val

    #         print('before train:', len(train_dataset))
    #         print('before valid:', len(valid_dataset))
    #         unlabeled_amount = int(len(train_dataset) * unlabeled_size)
    #         print('len of u:', unlabeled_amount)
    #         print('len of dataset:', len(train_dataset))


    #         #splitting the dataset into unlabeled and training datasets
    #         train_set, unlabeled_set = torch.utils.data.random_split(train_dataset, [
    #                     (len(train_dataset) - unlabeled_amount), 
    #                     unlabeled_amount
    #         ])

    #         if len(train_set)>0 and len(valid_dataset)>0 and len(unlabeled_set)>0:

    #             train_dataloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False, \
    #                                                     drop_last=True, num_workers=1, collate_fn=train_dataset.collate_function)

    #             unlabeled_dataloader = torch.utils.data.DataLoader(dataset=unlabeled_set, batch_size=batch_size, shuffle=False, \
    #                                                     drop_last=True, num_workers=1, collate_fn=train_dataset.collate_function)


    #             valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, \
    #                                                     drop_last=True, num_workers=1, collate_fn=valid_dataset.collate_function)
                
    #             print('train:', len(train_set))
    #             print('unlabeled:', len(unlabeled_set))
    #             print('valid:', len(valid_dataset))
            
    #             #main training loop
                
    #             print('\n')
    #             t = torch.cuda.get_device_properties(0).total_memory
    #             r = torch.cuda.memory_reserved(0) 
    #             al = torch.cuda.memory_allocated(0)
    #             f = r-al  # free inside reserved
    #             print('freeeee:', f)
            
    #             epoch_loss1 = mdl.train_model1(A_batch, train_dataloader, optimizer1, de_tokenizer, criterion, scheduler1)
    #             writer.add_scalar('Loss/model1', epoch_loss1, epoch)
    #             t = torch.cuda.get_device_properties(0).total_memory
    #             r = torch.cuda.memory_reserved(0) 
    #             al = torch.cuda.memory_allocated(0)
    #             f = r-al  # free inside reserved
    #             print('freeeee:', f)

    #             epoch_loss2 = mdl.train_model2(unlabeled_dataloader, optimizer2, de_tokenizer, criterion, scheduler2)# using the same training dataset for now.
    #             writer.add_scalar('Loss/model2', epoch_loss2, epoch)
    #             t = torch.cuda.get_device_properties(0).total_memory
    #             r = torch.cuda.memory_reserved(0) 
    #             al = torch.cuda.memory_allocated(0)
    #             f = r-al  # free inside reserved
    #             print('freeeee:', f)
    #             epoch_loss3, a_ind = mdl.val_model2( valid_dataloader, optimizer3, A, A_batch , de_tokenizer, criterion, scheduler3, a_ind)
    #             writer.add_scalar('Loss/val', epoch_loss3, epoch)
    #         if (inst_tr*(i+1))%100 == 0:
    #             print('saving model after'+str((inst_tr*(i+1)))+'instances')
    #             mdl.save_model(config['model_path'])

    #             # model1_path=config["model1"]["saved_model_path"]
    #             # model2_path=config["model2"]["saved_model_path"]
    #             # model1_path='./saved_models/model1'
    #             # model2_path='./saved_models/model2'
    #             # mdl=TranslationModel(device, batch_size, logging, model1_path, model2_path, config)
