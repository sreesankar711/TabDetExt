import { createSlice,createAsyncThunk } from "@reduxjs/toolkit";
import axios from "axios"
// const API_URL = 'https://valid-jaybird-lightly.ngrok-free.app/upload'
const API_URL = 'http://127.0.0.1:5000/upload'
// const API_URL = 'https://table-website-backend-zz7lsswbvq-el.a.run.app:8000/upload'
// const API_URL = 'https://tabledetection.us.to/upload'


export const sendtabDetect = createAsyncThunk('/sendtabDetect',
async(formData,thunkAPI) => {
    
    try{
        const response = await axios.post(API_URL, formData, {
            headers: {
              'Content-Type': 'multipart/form-data',
            },
            responseType : 'blob'
          });
        console.log('hey fulfilled')
       
        const val = URL.createObjectURL(response.data)
        return val
    }catch(error)
    {
        const message = (error.response && error.response.data
            &&error.response.data.message) || error.message
            ||error.toString()

            return thunkAPI.rejectWithValue(message)
    }
}
)

const tabDetectSlice = createSlice({
    name : 'tabDetect',
    initialState : {
        is_tab_detected : false,
        cropped_image : ""
    },
    reducers : {
        reset_tabDetect : state => {
            state.is_tab_detected = false
            URL.revokeObjectURL(state.cropped_image)
            state.cropped_image = ""
        }
    },
    extraReducers : (builder) =>{
        builder
        .addCase(sendtabDetect.fulfilled ,(state,action)=>{
            state.cropped_image = action.payload
            state.is_tab_detected = true
        })
        .addCase(sendtabDetect.pending ,(state,action)=>{
            state.cropped_image  = ""
            state.is_tab_detected = false
        })
        .addCase(sendtabDetect.rejected ,(state,action)=>{
            state.cropped_image = ""
            state.is_tab_detected = false
        })
    }
})

export const {reset_tabDetect} = tabDetectSlice.actions
export default tabDetectSlice.reducer