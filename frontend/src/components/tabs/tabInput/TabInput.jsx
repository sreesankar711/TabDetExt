import "./tabInput.css"
import { useDispatch } from "react-redux";
import { select, select_name } from "../../../features/imageSelected/imageSelectedSlice"
import { sendtabDetect } from "../../../features/tabDetect/tabDetectSlice"

function TabInput() {
    const dispatch = useDispatch()

    function file_change(e){
        let file = e.target.files[0]
        if (file) {
            console.log('hehe entered hereereere')
             let image_file = URL.createObjectURL(file)
             dispatch(select_name(file.name))
             dispatch(select(image_file))
             const formData = new FormData();
             formData.append('the_file', file);
             dispatch(sendtabDetect(formData))
         }
       
        e.target.value = null
    }
    return (
        <>

                <div className="container-tabInput" >
                    <div className="tabInput-left" >
                        <h1>Insert Images </h1> <h1>to extract</h1> <h1>tabular data</h1>
                    </div>
                    <div className="tabInput-right ">
                        <label htmlFor="file" className="shareOption btn-tabInput">
                            {/* <PermMedia htmlColor="tomato" className="shareIcon"/> */}
                            upload image

                            <input style={{ display: "none" }}
                                type="file" id="file" accept=".png,.jpeg,.jpg,.pdf"
                                onChange={file_change}
                            />
                        </label>

                    </div>

                </div>


        </>
    )
}
export default TabInput;