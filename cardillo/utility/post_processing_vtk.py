import numpy as np
import meshio
import os
import pickle


def post_processing(subsystem, t, q, filename, u=None, binary=True):
    # For subsystems with the same point data writes paraview PVD file collecting time and all vtk files,
    # see https://www.paraview.org/Wiki/ParaView/Data_formats#PVD_File_Format
    from xml.dom import minidom

    root = minidom.Document()

    vkt_file = root.createElement("VTKFile")
    vkt_file.setAttribute("type", "Collection")
    root.appendChild(vkt_file)

    collection = root.createElement("Collection")
    vkt_file.appendChild(collection)

    nodes = {}

    if u is None:
        u = np.zeros_like(q)

    for i, (ti, qi, ui) in enumerate(zip(t, q, u)):
        filei = filename + f"{i}.vtu"

        # write time step and file name in pvd file
        dataset = root.createElement("DataSet")
        dataset.setAttribute("timestep", f"{ti:0.6f}")
        dataset.setAttribute("file", filei)
        collection.appendChild(dataset)

        geom_points = np.array([]).reshape(0, 3)
        cells = []
        HigherOrderDegrees = []
        point_data = {}
        field_data = {}
        offset = 0
        nodes_ti = []

        for subsystemi in subsystem:
            (
                geom_pointsi,
                point_datai,
                cellsi,
                HigherOrderDegreesi,
            ) = subsystemi.post_processing_subsystem(
                ti, qi[subsystemi.qDOF], ui[subsystemi.uDOF], binary=binary
            )

            geom_points = np.append(geom_points, geom_pointsi, axis=0)

            # update cell type and global connectivity
            for k, (cell_type, connectivity) in enumerate(cellsi):
                cellsi[k] = (cell_type, connectivity + offset)
            cells.extend(cellsi)
            offset = cellsi[-1][-1][-1, -1] + 1

            HigherOrderDegrees.extend(HigherOrderDegreesi)

            # update point_data dictionary. For first subsystem generate dictionary
            for key in point_datai:
                if key in point_data:
                    point_data.update(
                        {key: np.append(point_data[key], point_datai[key], axis=0)})
                else:
                    point_data.update({key: point_datai[key]})

            # update field data
            field_data = {k: field_data.get(k, 0) + field_datai.get(k,0) for k in set(field_datai)}

            # update nodal positions
            nodes_ti.extend(subsystemi.centerline(qi, 5).T)
        nodes.update({i: nodes_ti})

        # write field data
        with open(filei.parent / f'field_data_{i}', mode="wb") as f:
            pickle.dump(field_data, f)


        # write vtk mesh using meshio
        meshio.write_points_cells(
            filei.parent / (filei.stem + '.vtu'),
            # os.path.splitext(os.path.basename(filei))[0] + '.vtu',
            geom_points,  # only export centerline as geometry here!
            cells,
            point_data=point_data,
            cell_data={"HigherOrderDegrees": HigherOrderDegrees},
            field_data=field_data,
            binary=binary
        )

    # write nodal positions for centerline points of beams to external file
    with open(filename.parent / 'nodes', mode="wb") as f:
        pickle.dump(nodes, f)

    # write pvd file
    xml_str = root.toprettyxml(indent="\t")
    with (filename.parent / (filename.stem + '.pvd')).open("w") as f:
        f.write(xml_str)
