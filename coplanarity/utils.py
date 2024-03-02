import pandas as pd
import numpy as np
import json
import glob
import os
import pickle


class Neuron(object):
    def __init__(self, file_path: str, scale: float = 1.0, mode: str = 't', eswc: bool = False, mirror: bool = False):
        self.fp = file_path
        self.scale = scale
        self.mode = mode
        self.eswc = eswc
        self.mirror = mirror

        self.swc = None
        self.soma = None
        self.nidHash = None
        self.indexChildren = None

        self.initialize()

    def __len__(self):
        return len(self.swc)

    def initialize(self):
        self.swc = self.read_swc()
        self.soma = self.get_soma()
        if self.mirror:
            z = self.soma[4]
            if z > 456 * 25 * self.scale / 2:
                self.swc[:, 4] = 456 * 25 * self.scale - self.swc[:, 4]
                self.soma[4] = 456 * 25 * self.scale - self.soma[4]

        self.nidHash = {}
        self.indexChildren = []
        for i in range(self.__len__()):
            self.nidHash[self.swc[i][0]] = i
            self.indexChildren.append([])
        for i in range(self.__len__()):
            pid = self.swc[i][6]
            idx = self.nidHash.get(pid)
            if idx is None: continue
            # print(self.swc[i],pid,idx,i)
            self.indexChildren[idx].append(i)

    def read_swc(self):
        swc_matrix = []
        with open(self.fp, 'r') as f:
            while True:
                linelist = []
                line = f.readline()
                if not line: break
                if line[0] == "#" or line[0] == 'i' or line[0] == "\n": continue
                line = line.strip("\n").strip(" ")
                if line.count("\t") >= line.count(" "):
                    str_split = "\t"
                elif line.count("\t") <= line.count(" "):
                    str_split = " "
                elem = line.split(str_split)
                # if self.mode == "t" or self.mode == 'T':
                #     pass
                # elif self.mode == "a" or self.mode == "A":  # 1s 2a 3d
                #     if elem[1] not in ['1', '2']:
                #         continue
                # elif self.mode == "d" or self.mode == "D":
                #     if elem[1] not in ['1', '3', '4']:
                #         continue
                for i in range(len(elem) if self.eswc else 7):
                    if i in [0, 1, 6]:
                        linelist.append(int(elem[i]))
                    elif i in [2, 3, 4]:
                        linelist.append(float(elem[i]) * self.scale)
                    else:
                        linelist.append(float(elem[i]))
                swc_matrix.append(linelist)
        swc_matrix = np.array(swc_matrix, dtype=object)

        return swc_matrix

    def get_soma(self):
        soma = np.array([])

        for i in range(len(self.swc)):
            if self.swc[i][1] == 1 and self.swc[i][6] == -1:
                soma = self.swc[i].copy()
                break
        if not soma.any():
            for i in range(len(self.swc)):
                if self.swc[i][6] == -1:
                    soma = self.swc[i].copy()
                    break
        # if not soma:
        #     for i in range(len(self.swc)):
        #         if self.swc[i][1] == 1:
        #             soma = self.swc[i]
        if not soma.any():
            print("no soma detected...")
        return soma


class Segment(object):
    def __init__(self):
        def __init__(self, file_path: str, scale: float = 1.0, mode: str = 't', eswc: bool = False,
                     mirror: bool = False):
            self.fp = file_path
            self.scale = scale
            self.mode = mode
            self.eswc = eswc
            self.mirror = mirror

            self.swc = None
            self.soma = None

            self.reset()


class MouseAnatomyTree:
    def __init__(self, treepath=r"E:\ZhixiYun\Projects\Neuron_Morphology_Table\Tables\tree.json", ):
        self.tree = []
        self.roughlist = ['Isocortex', 'OLF', 'HPF', 'CTXsp', 'STR', 'PAL',
                          'TH', 'HY', 'MB', 'P', 'MY', 'CBX', 'CBN', 'VS', 'fiber tracts']
        self.lutnametoid = {}
        self.lutidtoname = {}
        self.lutidtorgb = {}
        self.lutnametorough = {}

        self._id_index_hash = {}

        with open(treepath) as f:
            self.tree = json.load(f)

        for i, t in enumerate(self.tree):
            id_ = t["id"]
            self.lutnametoid[t["acronym"]] = id_
            self.lutidtoname[id_] = t["acronym"]
            self.lutidtorgb[id_] = t["rgb_triplet"]
            self._id_index_hash[id_] = i

        for t in self.tree:
            self.lutnametorough[t['acronym']] = t['acronym']
            for rough in self.roughlist:
                if self.lutnametoid.get(rough) in t['structure_id_path']:
                    self.lutnametorough[t['acronym']] = rough
                    break

    def _id_acronym_check(self, inp, inp_type: str):
        if inp_type not in ['id', 'acronym']:
            raise ValueError(f'invalid input type: {inp_type}, should be one of "id", "acronym".')
        if inp_type == 'id':
            if isinstance(inp, str):
                inp = self.lutnametoid.get(inp)
        elif inp_type == 'acronym':
            if isinstance(inp, int):
                inp = self.lutidtoname.get(inp)
        return inp

    def _ctlist_overlap_check(self, ctlist) -> bool:
        # check ctlist whether it has overlapping cell types in tree hierarchy
        ct_child_list = []
        for ct in ctlist:
            ct = self._id_acronym_check(ct, 'id')
            children = self.find_children_id(ct)

            if ct in ct_child_list:
                return True

            ct_child_list.extend(children)

        return False

    def find_children_id(self, id_):
        id_ = self._id_acronym_check(id_, 'id')
        idlist = []
        for t in self.tree:
            if id_ in t["structure_id_path"]:
                idlist.append(t['id'])
        if not idlist:
            idlist = [id_]
        return idlist

    def ccf_sort(self, ctlist):
        select_ct_sorted = []
        for item in self.tree:
            if item["acronym"] in ctlist:
                select_ct_sorted.append(item["acronym"])
        return select_ct_sorted

    def cortex_layer_to_upper(self, ctlist, SSp=False):
        newctlist = []
        _SSp_id = self.lutnametoid['SSp']
        for ct in ctlist:
            ct = self._id_acronym_check(ct, 'acronym')
            if self.lutnametorough.get(ct) == 'Isocortex':
                id_ = self.lutnametoid.get(ct)
                upper_id_list = self.tree[self._id_index_hash[id_]]['structure_id_path']
                if len(upper_id_list) >= 2:
                    upper_id = upper_id_list[-2]
                else:
                    upper_id = upper_id_list[-1]
                upper_acronym = self.lutidtoname.get(upper_id)

                if SSp:
                    if _SSp_id in upper_id_list:
                        upper_acronym = 'SSp'
                        newctlist.append(upper_acronym)
                        continue

                if ct[len(upper_acronym):] in ['1', '2/3', '4', '5', '6', '6a', '6b']:
                    newctlist.append(upper_acronym)
                else:
                    newctlist.append(ct)
            else:
                newctlist.append(ct)

        return newctlist

    def ctlist_to_given_ctlist(self, ctlist, given_ctlist, not_in_set_None=False):
        # bullshit code need to re-write
        overlap_flag = self._ctlist_overlap_check(given_ctlist)
        if overlap_flag:
            raise ValueError('given_ctlist has overlapping cell types')
        ctlist = np.asarray(ctlist)
        given_ctlist = np.asarray(given_ctlist)
        tmp_ctlist = []
        tmp_given_ctlist = []
        for ct in ctlist:
            ct = self._id_acronym_check(ct, 'id')
            tmp_ctlist.append(ct)
        tmp_ctlist = np.asarray(tmp_ctlist)

        out_arr = np.zeros(len(tmp_ctlist), dtype=object)
        if not_in_set_None:
            out_arr[out_arr == 0] = None
        else:
            out_arr = np.asarray(tmp_ctlist, dtype=object)

        for gct in given_ctlist:
            gct_children = self.find_children_id(gct)
            out_arr[np.isin(tmp_ctlist, gct_children)] = gct

        for i in range(len(out_arr)):
            out_arr[i] = self._id_acronym_check(out_arr[i], 'acronym')

        return out_arr


if __name__ == '__main1__':
    print()
