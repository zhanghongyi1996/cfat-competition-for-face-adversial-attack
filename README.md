This project mainly focused on the balance between how to find smaller area in photos and generate noise to attack the face recognition model as efficient as possible.

The time of noise generation is less than 100s per photo only with cpu.

Example code is listed here.


    for idname in range(1, 101):
        tool = pyfat_implement.PyFAT(N=10)                         # Do initalization(Import pyfat_implement.py)
        if args.device=='cuda':
            tool.set_cuda()                                        # Whether to use GPU
        tool.load('assets')                                        # Load models
        str_idname = "%03d"%idname
        iddir = osp.join('images', str_idname)
        att = osp.join(iddir, '0.png')
        vic = osp.join(iddir, '1.png')
        origin_att_img = cv2.imread(att)
        origin_vic_img = cv2.imread(vic)                           # Load attack and Victim images
        for turn in range(tool.size()):
            adv_img = tool.generate(origin_att_img, origin_vic_img, turn)          # Generate the graphes for attack
            save_name = '{}_fake_'.format(str_idname) + str(turn) + '_2.png'
            cv2.imwrite(save_dir + '/' + save_name, adv_img)
