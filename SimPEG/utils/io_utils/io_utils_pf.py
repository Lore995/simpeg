from __future__ import print_function
import numpy as np
from discretize.utils import mkvc
from SimPEG.utils.code_utils import deprecate_method


def read_magnetics_3d_ubc(obs_file):
    """
        Read and write UBC mag file format

        INPUT:
        :param fileName, path to the UBC obs mag file

        OUTPUT:
        :param survey
        :param M, magnetization orentiaton (MI, MD)
    """
    from SimPEG.potential_fields import magnetics
    from SimPEG import data

    fid = open(obs_file, "r")

    # First line has the inclination,declination and amplitude of B0
    line = fid.readline()
    B = np.array(line.split()[:3], dtype=float)

    # Second line has the magnetization orientation and a flag
    line = fid.readline()
    M = np.array(line.split()[:3], dtype=float)

    # Third line has the number of rows
    line = fid.readline()
    ndat = int(line.split()[0])

    # Pre-allocate space for obsx, obsy, obsz, data, uncert
    line = fid.readline()
    temp = np.array(line.split(), dtype=float)

    d = np.zeros(ndat, dtype=float)
    wd = np.zeros(ndat, dtype=float)
    locXYZ = np.zeros((ndat, 3), dtype=float)

    ii = 0
    while ii < ndat:

        temp = np.array(line.split(), dtype=float)
        if len(temp) > 0:
            locXYZ[ii, :] = temp[:3]

            if len(temp) > 3:
                d[ii] = temp[3]

                if len(temp) == 5:
                    wd[ii] = temp[4]
            ii += 1
        line = fid.readline()
    fid.close()

    if np.all(wd == 0.0):
        wd = None

    if np.all(d == 0.0):
        d = None

    rxLoc = magnetics.receivers.Point(locXYZ)
    srcField = magnetics.sources.SourceField([rxLoc], parameters=(B[2], B[0], B[1]))
    survey = magnetics.survey.Survey(srcField)
    data_object = data.Data(survey, dobs=d, standard_deviation=wd)

    return data_object


def write_magnetics_3d_ubc(filename, data_object):
    """
    writeUBCobs(filename,B,M,rxLoc,d,wd)

    Function writing an observation file in UBC-MAG3D format.

    INPUT
    filename    : Name of out file including directory
    survey
    flag          : dobs | dpred

    OUTPUT
    Obsfile

    Created on Dec, 27th 2015

    @author: dominiquef
    """
    survey = data_object.survey

    B = survey.source_field.parameters

    data = survey.source_field.receiver_list[0].locations

    if data_object.dobs is not None:
        data = np.c_[data, data_object.dobs]

    if data_object.standard_deviation is not None:
        data = np.c_[data, data_object.standard_deviation]

    head = (
        "%6.2f %6.2f %6.2f\n" % (B[1], B[2], B[0])
        + "%6.2f %6.2f %6.2f\n" % (B[1], B[2], 1)
        + "%i\n" % survey.nD
    )
    np.savetxt(
        filename, data, fmt="%e", delimiter=" ", newline="\n", header=head, comments=""
    )

    print("Observation file saved to: " + filename)


def read_gravity_3d_ubc(obs_file):
    """
    Read UBC grav file format

    INPUT:
    :param fileName, path to the UBC obs grav file
    :param ftype, 'dobs' 'dpred' 'survey'

    OUTPUT:
    :param survey

    """
    from SimPEG.potential_fields import gravity
    from SimPEG import data

    fid = open(obs_file, "r")

    # First line has the number of rows
    line = fid.readline()
    ndat = int(line.split()[0])

    # Pre-allocate space for obsx, obsy, obsz, data, uncert
    line = fid.readline()

    d = np.zeros(ndat, dtype=float)
    wd = np.zeros(ndat, dtype=float)
    locXYZ = np.zeros((ndat, 3), dtype=float)

    ii = 0
    while ii < ndat:

        temp = np.array(line.split(), dtype=float)
        if len(temp) > 0:
            locXYZ[ii, :] = temp[:3]
            if len(temp) > 3:
                d[ii] = temp[3]

                if len(temp) == 5:
                    wd[ii] = temp[4]

            ii += 1
        line = fid.readline()
    fid.close()

    if np.all(wd == 0.0):
        wd = None

    # UBC and SimPEG used opposite sign convention for
    # gravity data so must multiply by -1.
    if np.all(d == 0.0):
        d = None
    else:
        d *= -1.0

    rxLoc = gravity.receivers.Point(locXYZ)
    srcField = gravity.sources.SourceField([rxLoc])
    survey = gravity.survey.Survey(srcField)
    data_object = data.Data(survey, dobs=d, standard_deviation=wd)
    return data_object


def write_gravity_3d_ubc(filename, data_object):
    """
        Write UBC grav file format

        INPUT:
        :param: fileName, path to the UBC obs grav file
        :param: survey Gravity object
        :param: data array

    """
    survey = data_object.survey

    data = survey.source_field.receiver_list[0].locations

    # UBC and SimPEG use opposite sign for gravity data so
    # data are multiplied by -1.
    if data_object.dobs is not None:
        data = np.c_[data, -data_object.dobs]

    if data_object.standard_deviation is not None:
        data = np.c_[data, data_object.standard_deviation]

    head = "%i\n" % survey.nD
    np.savetxt(
        filename, data, fmt="%e", delimiter=" ", newline="\n", header=head, comments=""
    )

    print("Observation file saved to: " + filename)


def read_gravity_gradiometry_3d_ubc(obs_file, file_type):
    """
    Read UBC gravity gradiometry file format

    INPUT:
    :param fileName, path to the UBC obs gravity gradiometry file
    :param file_type, 'dobs' 'dpred' 'survey'

    OUTPUT:
    :param survey

    """
    if file_type not in ["survey", "dpred", "dobs"]:
        raise ValueError("file_type must be one of: 'survey', 'dpred', 'dobs'")

    from SimPEG.potential_fields import gravity
    from SimPEG import data

    fid = open(obs_file, "r")

    # First line has components. Extract components
    line = fid.readline()
    line = line.split("=")[1].split("!")[0].split("\n")[0]
    line = line.replace(",", " ").split(" ")  # UBC uses ',' or ' ' as deliminator
    components = [s for s in line if len(s) > 0]  # Remove empty string
    factor = np.zeros(len(components))

    # Convert component types from UBC to SimPEG
    ubc_types = ["xx", "xy", "xz", "yy", "yz", "zz"]
    simpeg_types = ["gyy", "gxy", "gyz", "gxx", "gxz", "gzz"]
    factor_list = [1.0, 1.0, -1.0, 1.0, -1.0, 1.0]

    for ii in range(0, len(components)):
        k = ubc_types.index(components[ii])
        factor[ii] = factor_list[k]
        components[ii] = simpeg_types[k]

    # Second Line has number of locations
    line = fid.readline()
    ndat = int(line.split()[0])

    # Pre-allocate space for obsx, obsy, obsz, data, uncert
    line = fid.readline()

    locXYZ = np.zeros((ndat, 3), dtype=float)
    if file_type == "survey":
        d = None
        wd = None
    elif file_type == "dpred":
        d = np.zeros((ndat, len(components)), dtype=float)
        wd = None
    else:
        d = np.zeros((ndat, len(components)), dtype=float)
        wd = np.zeros((ndat, len(components)), dtype=float)

    ii = 0
    while ii < ndat:

        temp = np.array(line.split(), dtype=float)
        locXYZ[ii, :] = temp[:3]

        if file_type == "dpred":
            d[ii, :] = factor * temp[3:]

        elif file_type == "dobs":
            d[ii, :] = factor * temp[3::2]
            wd[ii, :] = temp[4::2]

        ii += 1
        line = fid.readline()
    fid.close()

    # Turn into vector. For multiple components, SimPEG orders by rows
    if d is not None:
        d = mkvc(d.T)
    if wd is not None:
        wd = mkvc(wd.T)

    rxLoc = gravity.receivers.Point(locXYZ, components=components)
    srcField = gravity.sources.SourceField([rxLoc])
    survey = gravity.survey.Survey(srcField)
    data_object = data.Data(survey, dobs=d, standard_deviation=wd)
    return data_object


def write_gravity_gradiometry_3d_ubc(filename, data_object):
    """
        Write UBC gravity gradiometry file format

        INPUT:
        :param: fileName, path to the UBC obs grav file
        :param: survey Gravity object
        :param: data array

    """
    survey = data_object.survey

    # Convert component types from UBC to SimPEG
    components = list(survey.components.keys())
    n_comp = len(components)
    factor = np.ones(n_comp)

    ubc_types = ["xx", "xy", "xz", "yy", "yz", "zz"]
    simpeg_types = ["gyy", "gxy", "gyz", "gxx", "gxz", "gzz"]
    factor_list = [1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
    for ii in range(0, len(components)):
        k = simpeg_types.index(components[ii])
        factor[ii] = factor_list[k]
        components[ii] = ubc_types[k]

    components = ",".join(components)

    data = survey.source_field.receiver_list[0].locations
    n_loc = np.shape(data)[0]

    if np.any(data_object.standard_deviation != 0):
        for ii in range(0, n_comp):
            data = np.c_[
                data,
                factor[ii] * data_object.dobs[ii::n_comp],
                data_object.standard_deviation[ii::n_comp],
            ]

    elif np.any(data_object.dobs != 0):
        for ii in range(0, n_comp):
            data = np.c_[data, factor[ii] * data_object.dobs[ii::n_comp]]

    head = ("datacomp=%s\n" % components) + ("%i" % n_loc)

    np.savetxt(
        filename, data, fmt="%e", delimiter=" ", newline="\n", header=head, comments=""
    )

    print("Observation file saved to: " + filename)


# ======================================================
# 				Depricated Methods
# ======================================================


readUBCmagneticsObservations = deprecate_method(
    read_magnetics_3d_ubc, "readUBCmagneticsObservations", removal_version="0.15.0"
)
writeUBCmagneticsObservations = deprecate_method(
    write_magnetics_3d_ubc, "writeUBCmagneticsObservations", removal_version="0.15.0"
)
readUBCgravityObservations = deprecate_method(
    read_gravity_3d_ubc, "readUBCgravityObservations", removal_version="0.15.0"
)
writeUBCgravityObservations = deprecate_method(
    write_gravity_3d_ubc, "writeUBCgravityObservations", removal_version="0.15.0"
)
readUBCgravitygradiometryObservations = deprecate_method(
    read_gravity_gradiometry_3d_ubc,
    "readUBCgravitygradiometryObservations",
    removal_version="0.15.0",
)
writeUBCgravitygradiometryObservations = deprecate_method(
    write_gravity_gradiometry_3d_ubc,
    "writeUBCgravitygradiometryObservations",
    removal_version="0.15.0",
)
