import { configureStore } from '@reduxjs/toolkit'
import imageSelectedReducer from "../features/imageSelected/imageSelectedSlice"
import tabReducer from '../features/tabDetect/tabDetectSlice'

export const store = configureStore({
  reducer: {
    imageSelected : imageSelectedReducer,
    tabDetect : tabReducer
  }
})