import styles from "./tableDetect.module.css"
import { useSelector } from "react-redux";
import { useDispatch } from "react-redux";
import { reset_select } from "../../../features/imageSelected/imageSelectedSlice";
import { reset_tabDetect } from "../../../features/tabDetect/tabDetectSlice";

function TableDetect({updates}) {
    const cropped_image = useSelector(state => state.tabDetect.cropped_image)
    const dispatch = useDispatch()

    function download_fun() {
        const a = document.createElement('a');
        a.href = cropped_image;
        a.download = 'output_final.zip';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }

    function go_first_page(){
        dispatch(reset_select())
        dispatch(reset_tabDetect())
        updates(0)
    }

    return (
        <>
            <div className={styles.container} >
                <h1>Excel file generated</h1>
                <div className={styles.buttonContainer}>
                    <button className={styles.button} onClick={download_fun}>download</button>
                    <button className={styles.button} onClick={go_first_page}> upload new image</button>
                </div>

            </div>
        </>
    )
}
export default TableDetect;