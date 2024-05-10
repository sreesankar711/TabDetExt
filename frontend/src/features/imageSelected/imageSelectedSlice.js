import { createSlice } from "@reduxjs/toolkit";


const imageSelectedSlice = createSlice({
    name : 'imageSelected',
    initialState :{
        selected : false,
        image : null,
        name : ''
    },
    reducers : {
        select : (state,action) => {
            state.selected = true
            state.image = action.payload
        },
        select_name :(state,action)=> {
            state.name = action.payload
        },
        reset_select : (state) => {
            state.selected = false
            state.image = null
            state.name = ''
        }
    }
})

export const {select,select_name,reset_select} = imageSelectedSlice.actions

export default imageSelectedSlice.reducer