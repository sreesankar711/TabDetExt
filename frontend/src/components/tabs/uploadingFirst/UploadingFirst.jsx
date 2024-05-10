import styles from "./Uploading.module.css"
import { useSelector } from "react-redux";
import FadeLoader from "react-spinners/FadeLoader"
import { useEffect } from "react";
function UploadingFirst({index}) {

    const {selected,image,name} = useSelector(state => state.imageSelected)

    useEffect(()=>{
        if(selected){
            console.log(name)
        }
    },[image,selected,name])
    return(
        <>
        <div className={styles.container} style={{display:(selected)?"flex":"none"}}>
            
            <div className ={styles.left} style={{width : (name.split('.').pop()==='pdf')?"100%":"50%"}}><h1>Uploading...</h1>
            {index===1 &&(<FadeLoader color="white" className={styles.ring}/>)}
            </div>
            <div  className = {styles.right} style={{backgroundImage :`url(${image})`, display :(name.split('.').pop()==='pdf')?"none":"block"}}></div>
        </div>
        </>
    )
}
export default UploadingFirst;