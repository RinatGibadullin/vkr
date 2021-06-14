import io
import struct
import enum
import mutagen

FMT_VERSION = 'BB'


class EntryType(enum.IntEnum):
    INVALID = 0
    CUE = 1
    LOOP = 3


def serato32encode(data):
    """Encode 12 byte plain text into 4 byte Serato binary format."""
    a, b, c = struct.unpack('BBB', data)
    z = c & 0x7F
    y = ((c >> 7) | (b << 1)) & 0x7F
    x = ((b >> 6) | (a << 2)) & 0x7F
    w = (a >> 5)
    return bytes(bytearray([w, x, y, z]))


def serato32decode(data):
    """Decode 4 byte Serato binary format into 12 byte plain text."""
    w, x, y, z = struct.unpack('BBBB', data)
    c = (z & 0x7F) | ((y & 0x01) << 7)
    b = ((y & 0x7F) >> 1) | ((x & 0x03) << 6)
    a = ((x & 0x7F) >> 2) | ((w & 0x07) << 5)
    return struct.pack('BBB', a, b, c)


class Entry(object):
    FMT = '>B4sB4s6s4sBB'
    FIELDS = ('start_position_set', 'start_position', 'end_position_set',
              'end_position', 'field5', 'color', 'type', 'is_locked')

    def __init__(self, *args):
        assert len(args) == len(self.FIELDS)
        for field, value in zip(self.FIELDS, args):
            setattr(self, field, value)

    def __repr__(self):
        return '{name}({data})'.format(
            name=self.__class__.__name__,
            data=', '.join('{}={!r}'.format(name, getattr(self, name))
                           for name in self.FIELDS))

    @classmethod
    def load(cls, data):
        info_size = struct.calcsize(cls.FMT)
        info = struct.unpack(cls.FMT, data[:info_size])
        entry_data = []

        start_position_set = None
        end_position_set = None
        for field, value in zip(cls.FIELDS, info):
            if field == 'start_position_set':
                assert value in (0x00, 0x7F)
                value = value != 0x7F
                start_position_set = value
            elif field == 'end_position_set':
                assert value in (0x00, 0x7F)
                value = value != 0x7F
                end_position_set = value
            elif field == 'start_position':
                assert start_position_set is not None
                if start_position_set:
                    byte_value = serato32decode(value).rjust(4, b'\x00')
                    value = struct.unpack(
                        '>I', byte_value)[0]
                else:
                    value = None
            elif field == 'end_position':
                assert end_position_set is not None
                if end_position_set:
                    value = struct.unpack(
                        '>I', serato32decode(value).rjust(4, b'\x00'))[0]
                else:
                    value = None
            elif field == 'color':
                value = serato32decode(value)
            elif field == 'type':
                value = EntryType(value)
            entry_data.append(value)

        return cls(*entry_data)

    def dump(self):
        entry_data = []
        for field in self.FIELDS:
            value = getattr(self, field)
            if field == 'start_position_set':
                value = 0x7F if not value else 0x00
            elif field == 'end_position_set':
                value = 0x7F if not value else 0x00
            elif field == 'color':
                value = serato32encode(value)
            elif field == 'start_position':
                if value is None:
                    value = b'\x00\x7f\x7f\x7f\x7f'
                else:
                    value = serato32encode(struct.pack('>I', value)[1:])
            elif field == 'end_position':
                if value is None:
                    value = b'\x00\x7f\x7f\x7f\x7f'
                else:
                    value = serato32encode(struct.pack('>I', value)[1:])
            elif field == 'type':
                value = int(value)
            entry_data.append(value)
        return struct.pack(self.FMT, *entry_data)


class Color(Entry):
    FMT = '>4s'
    FIELDS = ('color',)


def parse(fp):
    assert struct.unpack(FMT_VERSION, fp.read(2)) == (0x02, 0x05)

    num_entries = struct.unpack('>I', fp.read(4))[0]
    for i in range(num_entries):
        entry_data = fp.read(0x16)
        assert len(entry_data) == 0x16

        entry = Entry.load(entry_data)
        yield entry

    yield Color.load(fp.read())


def dump(new_entries):
    data = struct.pack(FMT_VERSION, 0x02, 0x05)
    num_entries = len(new_entries) - 1
    data += struct.pack('>I', num_entries)
    for entry_data in new_entries:
        data += entry_data.dump()
    return data

COLORS = [b'\xcc\x00D',b'\x88\xcc\x00', b'\x88\x00\xcc', b'\xcc\xcc\x00', b'\xccD\x00', b'\xccD\x00', b'\xccD\x00', b'\xccD\x00', b'\xccD\x00']

def create_serato_markers(est_times, est_labels, audio_filename):

    tagfile = mutagen.File(audio_filename)
    if tagfile is not None:
        try:
            data = tagfile['GEOB:Serato Markers_'].data
        except KeyError:
            print('File is missing "GEOB:Serato Markers_" tag')
            return 1

    entries = list(parse(io.BytesIO(data)))

    for idx, time in enumerate(est_times):
        entry = Entry(True, time, False, None, b'\x00\x7f\x7f\x7f\x7f\x7f', COLORS[est_labels[idx]], EntryType.CUE, 0)
        entries[idx] = entry

    # entry = Entry(True, 50, False, None, b'\x00\x7f\x7f\x7f\x7f\x7f', b'\xcc\x00\x00', EntryType.CUE, 0)
    # empty_entry = Entry(False, None, False, None, b'\x00\x7f\x7f\x7f\x7f\x7f', b'\x00\x00\x00', EntryType.LOOP, 0)
    # empty_entries = [empty_entry] * 13
    # new_entries = [
    #     entry
    # ]
    # new_entries.extend(empty_entries)
    # entries[0] = entry
    new_data = dump(entries)
    if tagfile is not None:
        tagfile['GEOB:Serato Markers_'] = mutagen.id3.GEOB(
            encoding=0,
            mime='application/octet-stream',
            desc='Serato Markers_',
            data=new_data,
        )
        tagfile.save()