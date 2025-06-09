/**
 * Converts an image URL to a Base64 data URL (data:image/png;base64,...)
 * @param url - The image URL to convert
 * @returns Promise that resolves with the data URL string
 */
export async function urlToDataURL(url: string): Promise<string> {
    // If it's already a base64 data URI, return as-is
    if (url.startsWith('data:')) {
      return url;
    }
  
    const response = await fetch(url);
    const blob = await response.blob();
  
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
  
      reader.onloadend = () => {
        if (typeof reader.result === 'string') {
          resolve(reader.result); // this will be the data:image/... string
        } else {
          reject(new Error('Failed to convert image to Base64'));
        }
      };
  
      reader.onerror = (err) => reject(err);
      
      reader.readAsDataURL(blob);
    });
  }
  