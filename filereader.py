import mmap
import os


class FileReader:
    def __init__(self, path):
        self.file = os.open(path, os.O_RDONLY)
        self.mm = mmap.mmap(self.file, 0, prot=mmap.PROT_READ)
        self.idxes = []
        line = b'init'

        print("FileReader: indexing...")
        while line != b'':
            self.idxes.append(self.mm.tell())
            line = self.mm.readline()
        self.idxes.pop()
        print("Done!")

    def __getitem__(self, item):
        self.mm.seek(self.idxes[item])
        return self.mm.readline().decode("ascii", "ignore")

    def __len__(self):
        return len(self.idxes)


if __name__ == "__main__":
    import sys
    reader = FileReader(sys.argv[1])
    print("Len: {}".format(len(reader)))

    import code
    code.interact(local=locals())
