#ifndef Stk_UnitTest_ElemDeath_Utils
#define Stk_UnitTest_ElemDeath_Utils

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_topology/topology.hpp>

namespace ElementDeathUtils
{

inline void deactivate_elements(const stk::mesh::EntityVector &deactivated_elems, stk::mesh::BulkData &bulkData, stk::mesh::Part& active)
{
    active.set_primary_entity_rank(stk::topology::ELEM_RANK);

    bulkData.modification_begin();

    for(size_t i = 0; i < deactivated_elems.size(); ++i)
    {
        bulkData.change_entity_parts(deactivated_elems[i], stk::mesh::PartVector(), stk::mesh::PartVector(1, &active));
    }

    bulkData.modification_end();

    active.set_primary_entity_rank(stk::topology::INVALID_RANK);
}

inline stk::mesh::Entity get_face_between_element_ids(stk::mesh::ElemElemGraph& graph, stk::mesh::BulkData& bulkData, stk::mesh::EntityId elem1Id, stk::mesh::EntityId elem2Id)
{
    stk::mesh::Entity elem1 = bulkData.get_entity(stk::topology::ELEM_RANK, elem1Id);
    stk::mesh::Entity elem2 = bulkData.get_entity(stk::topology::ELEM_RANK, elem2Id);

    bool isElem1LocallyOwnedAndValid = bulkData.is_valid(elem1) && bulkData.bucket(elem1).owned();
    bool isElem2LocallyOwnedAndValid = bulkData.is_valid(elem2) && bulkData.bucket(elem2).owned();

    stk::mesh::Entity face_between_elem1_and_elem2;

    if(isElem1LocallyOwnedAndValid && isElem2LocallyOwnedAndValid)
    {
        int side = graph.get_side_from_element1_to_locally_owned_element2(elem1, elem2);
        EXPECT_TRUE(side != -1);
        face_between_elem1_and_elem2 = stk::mesh::impl::get_side_for_element(bulkData, elem1, side);
    }
    else if(isElem1LocallyOwnedAndValid)
    {
        int side = graph.get_side_from_element1_to_remote_element2(elem1, elem2Id);
        EXPECT_TRUE(side != -1);
        face_between_elem1_and_elem2 = stk::mesh::impl::get_side_for_element(bulkData, elem1, side);
    }
    else if(isElem2LocallyOwnedAndValid)
    {
        int side = graph.get_side_from_element1_to_remote_element2(elem2, elem1Id);
        EXPECT_TRUE(side != -1);
        face_between_elem1_and_elem2 = stk::mesh::impl::get_side_for_element(bulkData, elem2, side);
    }
    return face_between_elem1_and_elem2;
}

} // end namespace


#endif
