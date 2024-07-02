import numpy as np
import pickle

def divide_in_batches(positiveTrainList, nsteps, template=True, templateSize=1, frames_ids=None):
    setMaximum = 20

    genuineBatches = []
    frames_ids_batches = []

    batch_per_track = []
    for i in range(len(positiveTrainList)):
        batch_per_track.append(0)
    for i in range(nsteps):
        batch_per_track[i % len(positiveTrainList)] += 1

    # print(len(positiveTrainList))
    # print(batch_per_track)

    # Add or not the template
    if (template):
        '''
        with open('scores_alt.csv', 'a+', encoding='utf-8') as outfile:
            for val in positiveTrainList[0][:templateSize, :] / np.linalg.norm(positiveTrainList[0][:templateSize, :]):
                for dato in val:
                    outfile.write('%1.6f \n' % dato)
        '''
        #genuineBatches.append(positiveTrainList[0][:templateSize, :] / np.linalg.norm(positiveTrainList[0][:templateSize, :]))
        genuineBatches.append(positiveTrainList[0][:templateSize,:])
        if not frames_ids == None:
            frames_ids_batches.append(frames_ids[0][:templateSize])
        firstTime = True
    else:
        firstTime = False

    itime = 0

    # print(len(positiveTrainList))
    # print(len(frames_ids))

    for k, positiveTrain in enumerate(positiveTrainList):
        if firstTime:
            ini = templateSize
            firstTime = False
        else:
            ini = 0
        # Compute number of elements per step
        nbatchList = []
        for i in range(batch_per_track[itime]):
            nbatchList.append(0)
        for i in range(ini, positiveTrain.shape[0]):
            nbatchList[(i - ini) % batch_per_track[itime]] += 1
        # print(nbatchList)
        candidates = []
        temp_frames_ids = []

        ibatch = 0
        iframe = 0
        for i in range(ini, positiveTrain.shape[0]):

            iframe += 1

            # # Take the candidate
            # print(positiveTrain[i,:].reshape(1,-1).shape)
            # input()
            if (len(candidates) < setMaximum):
                '''
                with open('scores_alt.csv', 'a+', encoding='utf-8') as outfile:
                    for val in (positiveTrain[i, :]/np.linalg.norm(positiveTrain[i,:])).reshape(1, -1).tolist()[0]:
                        outfile.write('%1.6f \n' % val)
                '''
                #candidates.append((positiveTrain[i, :]/np.linalg.norm(positiveTrain[i,:])).reshape(1, -1).tolist()[0])
                candidates.append(positiveTrain[i, :].reshape(1, -1).tolist()[0])
                if not frames_ids == None:
                    # Le selecciono los frames asociados al individuo k
                    temp_frames_ids.append(frames_ids[k][i])

            # Determines de batch to add a new element
            if iframe == nbatchList[ibatch]:

                genuineBatches.append(np.array(candidates, dtype=np.float32))
                if not frames_ids == None:
                    frames_ids_batches.append(temp_frames_ids)
                    temp_frames_ids = []

                iframe = 0

                candidates = []

                ibatch += 1

        itime += 1

    return genuineBatches, frames_ids_batches


def load_general_div_nested(fileList, marklocation=-2, marklocation2=-3, maxElem=0):
    i = 1

    imUser = []
    camUser = []
    listFeatures = []
    database_ids = []
    cam_ids = []
    im_ids = []
    frames_ids = []

    mark = None
    with open(fileList, "r") as l:
        for line in l:
            sPath = line.split("/")
            if mark == None:
                print('<-', end='')
                mark = sPath[marklocation]  # User
                # print(mark, i-1)
                mark2 = sPath[marklocation2]  # seq
                # print(mark, mark2)

            if mark == sPath[marklocation]:

                if not (mark2 == sPath[marklocation2]):
                    camUser.append(np.array(imUser))
                    cam_ids.append(im_ids)
                    imUser = []
                    im_ids = []
                    sPath = line.split("/")
                    mark2 = sPath[marklocation2]
                    # print(mark, mark2, line, line.find(mark))

                with open('/mnt/cesar/FaceProc1/FaceProc'+line[1:-1], "rb") as feature:
                    imUser.append(np.fromfile(feature, dtype=np.float32))
                    im_ids.append(line[11:-5]+'.jpg')#sPath[-1][:-5])

            else:
                print('-', end='')
                i += 1

                camUser.append(np.array(imUser))
                cam_ids.append(im_ids)

                listFeatures.append(camUser)
                frames_ids.append(cam_ids)

                camUser = []
                imUser = []

                im_ids = []
                cam_ids = []

                sPath = line.split("/")
                database_ids.append(mark)

                mark = sPath[marklocation]
                # print(mark, i-1)
                mark2 = sPath[marklocation2]
                with open('/mnt/cesar/FaceProc1/FaceProc'+line[1:-1], "rb") as feature:
                    imUser.append(np.fromfile(feature, dtype=np.float32))
                    im_ids.append(line[11:-5]+'.jpg')#sPath[-1][:-5])

            if i == maxElem:
                break

    # print(i)
    print('>')

    camUser.append(np.array(imUser))
    cam_ids.append(im_ids)

    listFeatures.append(np.array(camUser))
    frames_ids.append(cam_ids)
    database_ids.append(mark)

    # for i in range(len(listFeatures)): print(len(listFeatures[i]))
    imUser = []
    camUser = []

    return listFeatures, database_ids, frames_ids


def load_data_to_process():
    IoItrainList, _,_ =load_general_div_nested('/mnt/cesar/FaceProc1/FaceProc/Experiments/FACE-COX/users-frames.txt', -2, -3)
    IoItrain = []
    for ll, IoI in enumerate(IoItrainList):
        temp, temp_ids = divide_in_batches(IoI, 9, template=True, templateSize=10)
        # Se añaden los lotes de individuos de interés
        IoItrain.append(temp)
    del temp
    print(len(IoItrain))
    file_features = open('FaceCOX_RN100_512D_full_splitted.obj', 'wb')
    pickle.dump(IoItrain, file_features)

if __name__ == '__main__':
    load_data_to_process()